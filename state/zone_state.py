# 파일: ~/parking_project/pie/pie/state/zone_state.py

# ============================================================
# [Phase 3] 구역 상태 머신
# 빈 구역 스냅샷 기준으로 픽셀 비교해서 차량 유무 판단
# ============================================================

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from config.settings import (
    STILL_PIXEL_THRESHOLD,
    STILL_SECONDS_REQUIRED,
    EXIT_TIMEOUT_SECONDS,
    RECHECK_INTERVAL_SEC,
    AISLE_STILL_SECONDS,
    AISLE_ZONE_PREFIX,
    AISLE_ZONES,
    PIXEL_DIFF_THRESHOLD,
    PIXEL_CHECK_OCCUPIED,
    PIXEL_LIGHTING_CHANGE_THRESHOLD,
    PIXEL_LIGHTING_UPDATE_INTERVAL,
)

NO_CAR_FRAMES_BEFORE_TIMEOUT = 15
EXIT_COOLDOWN_SECONDS = 5.0


class ZoneStatus(Enum):
    EMPTY    = "empty"
    OCCUPIED = "occupied"
    TIMEOUT  = "timeout"


class ParkStatus(Enum):
    NORMAL      = "normal"
    DOUBLE_PARK = "double_park"
    MULTI_ZONE  = "multi_zone"
    AISLE_BLOCK = "aisle_block"


class PlateStatus(Enum):
    PENDING    = "pending"
    CONFIRMED  = "confirmed"
    NULL       = "null"
    UNREADABLE = "unreadable"


@dataclass
class ZoneState:
    name:             str
    status:           ZoneStatus  = ZoneStatus.EMPTY
    plate:            str | None  = None
    plate_status:     PlateStatus = PlateStatus.PENDING
    park_status:      ParkStatus  = ParkStatus.NORMAL

    last_foot:    tuple = field(default_factory=lambda: (0, 0))
    still_since:  float = 0.0
    is_still:     bool  = False

    timeout_start:     float = 0.0
    last_recheck_time: float = 0.0

    double_park_suspected: bool       = False
    linked_zone:           str | None = None
    entry_time:            float      = 0.0

    empty_snap:           np.ndarray | None = field(default=None, repr=False)

    last_lighting_check:  float = 0.0
    last_mean_brightness: float = -1.0

    no_car_count: int = 0
    exit_cooldown_until: float = 0.0


class ParkingStateMachine:
    def __init__(self, zone_names: list[str]):
        self.zones: dict[str, ZoneState] = {
            name: ZoneState(name=name) for name in zone_names
        }

    def save_empty_snap(self, zone_name: str, zone_crop: np.ndarray):
        zone = self.zones.get(zone_name)
        if zone is None:
            return
        if zone.status != ZoneStatus.EMPTY:
            return
        zone.empty_snap           = zone_crop.copy()
        zone.last_mean_brightness = float(
            cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY).mean()
        )
        zone.last_lighting_check  = time.time()

    def update(self, zone_name: str,
               virtual_foot: tuple | None,
               all_cars_in_zone: list[dict],
               plate_visible: bool = True,
               zone_crop: np.ndarray | None = None) -> dict | None:
        zone = self.zones.get(zone_name)
        if zone is None:
            return None

        now = time.time()
        is_aisle  = zone_name.startswith(AISLE_ZONE_PREFIX)
        still_req = AISLE_STILL_SECONDS if is_aisle else STILL_SECONDS_REQUIRED

        if zone.status == ZoneStatus.OCCUPIED:
            zone.double_park_suspected = (
                len(all_cars_in_zone) >= 2 or not plate_visible
            )

        if (zone.status == ZoneStatus.EMPTY
                and zone_crop is not None
                and zone.empty_snap is not None):
            self._update_snap_if_lighting_changed(zone, zone_crop, now)

        yolo_found = virtual_foot is not None

        pixel_has_car = False
        if not yolo_found and zone.status in (ZoneStatus.OCCUPIED, ZoneStatus.TIMEOUT):
            pixel_has_car = self._pixel_check(zone, zone_crop)

        car_present = yolo_found or pixel_has_car

        # ── EMPTY ─────────────────────────────────────────
        if zone.status == ZoneStatus.EMPTY:
            if zone.exit_cooldown_until > 0 and now < zone.exit_cooldown_until:
                return None

            if zone.exit_cooldown_until > 0 and now >= zone.exit_cooldown_until:
                zone.exit_cooldown_until = 0.0

            if yolo_found:
                return self._handle_entry(zone, virtual_foot, now, still_req)

        # ── OCCUPIED ──────────────────────────────────────
        elif zone.status == ZoneStatus.OCCUPIED:
            if car_present:
                zone.timeout_start = 0.0
                zone.no_car_count  = 0

                if not yolo_found and pixel_has_car:
                    if zone_crop is not None:
                        if (now - zone.last_lighting_check
                                > PIXEL_LIGHTING_UPDATE_INTERVAL * 2):
                            zone.empty_snap           = zone_crop.copy()
                            zone.last_mean_brightness = float(
                                cv2.cvtColor(
                                    zone_crop, cv2.COLOR_BGR2GRAY
                                ).mean()
                            )
                            zone.last_lighting_check  = now
                return None
            else:
                zone.no_car_count += 1
                if zone.no_car_count >= NO_CAR_FRAMES_BEFORE_TIMEOUT:
                    zone.no_car_count  = 0
                    zone.status        = ZoneStatus.TIMEOUT
                    zone.timeout_start = now
                return None

        # ── TIMEOUT ───────────────────────────────────────
        elif zone.status == ZoneStatus.TIMEOUT:
            if car_present:
                zone.status        = ZoneStatus.OCCUPIED
                zone.timeout_start = 0.0
                zone.no_car_count  = 0
                return None

            if zone.timeout_start == 0.0:
                zone.timeout_start = now
                return None

            elapsed = now - zone.timeout_start
            if elapsed >= EXIT_TIMEOUT_SECONDS:
                old_plate        = zone.plate
                old_plate_status = zone.plate_status
                old_entry_time   = zone.entry_time
                old_status       = zone.park_status
                old_linked       = zone.linked_zone

                self._reset_zone(zone, zone_crop)
                zone.exit_cooldown_until = now + EXIT_COOLDOWN_SECONDS

                if old_linked:
                    linked_zone = self.zones.get(old_linked)
                    if linked_zone and linked_zone.status in (
                        ZoneStatus.OCCUPIED, ZoneStatus.TIMEOUT
                    ):
                        self._reset_zone(linked_zone, zone_crop)
                        linked_zone.exit_cooldown_until = now + EXIT_COOLDOWN_SECONDS

                return {
                    "type":         "exit",
                    "zone":         zone_name,
                    "plate":        old_plate,
                    "plate_status": old_plate_status.value,
                    "entry_time":   old_entry_time,
                    "park_status":  old_status.value,
                    "linked_zone":  old_linked,
                    "timestamp":    now,
                }

        return None

    def _pixel_check(self, zone: ZoneState,
                     zone_crop: np.ndarray | None) -> bool:
        if zone.empty_snap is None or zone_crop is None:
            return False

        try:
            snap = zone.empty_snap
            curr = zone_crop

            if snap.shape != curr.shape:
                curr = cv2.resize(curr, (snap.shape[1], snap.shape[0]))

            snap_gray = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            diff          = cv2.absdiff(snap_gray, curr_gray)
            changed_ratio = np.sum(diff > PIXEL_DIFF_THRESHOLD) / diff.size

            return changed_ratio > PIXEL_CHECK_OCCUPIED

        except Exception:
            return False

    def _update_snap_if_lighting_changed(self, zone, zone_crop, now):
        if zone.status != ZoneStatus.EMPTY:
            return
        if now - zone.last_lighting_check < PIXEL_LIGHTING_UPDATE_INTERVAL:
            return
        zone.last_lighting_check = now

        try:
            gray        = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY)
            mean_bright = float(gray.mean())

            if zone.last_mean_brightness < 0:
                zone.last_mean_brightness = mean_bright
                return

            if abs(mean_bright - zone.last_mean_brightness) \
                    > PIXEL_LIGHTING_CHANGE_THRESHOLD:
                if zone.status == ZoneStatus.EMPTY:
                    zone.empty_snap           = zone_crop.copy()
                    zone.last_mean_brightness = mean_bright
            else:
                zone.last_mean_brightness = mean_bright
        except Exception:
            pass

    def _handle_entry(self, zone, foot, now, still_req) -> dict | None:
        dx = abs(foot[0] - zone.last_foot[0])
        dy = abs(foot[1] - zone.last_foot[1])
        zone.last_foot = foot

        if max(dx, dy) < STILL_PIXEL_THRESHOLD:
            if not zone.is_still:
                zone.is_still    = True
                zone.still_since = now
            elif now - zone.still_since >= still_req:
                zone.status       = ZoneStatus.OCCUPIED
                zone.plate        = None
                zone.plate_status = PlateStatus.PENDING
                zone.entry_time   = now
                zone.no_car_count = 0
                zone.park_status  = (
                    ParkStatus.AISLE_BLOCK
                    if zone.name.startswith(AISLE_ZONE_PREFIX)
                    else ParkStatus.NORMAL
                )
                zone.last_recheck_time = now
                return {
                    "type":         "entry",
                    "zone":         zone.name,
                    "plate":        None,
                    "plate_status": PlateStatus.PENDING.value,
                    "entry_time":   now,
                    "park_status":  zone.park_status.value,
                    "timestamp":    now,
                }
        else:
            zone.is_still    = False
            zone.still_since = 0.0
        return None

    def set_plate(self, zone_name: str, plate: str | None):
        from ocr.reader import PLATE_UNREADABLE
        zone = self.zones.get(zone_name)
        if zone is None or zone.status != ZoneStatus.OCCUPIED:
            return

        if plate == PLATE_UNREADABLE:
            zone.plate        = None
            zone.plate_status = PlateStatus.UNREADABLE
        elif plate is None:
            zone.plate_status = PlateStatus.NULL
        else:
            zone.plate        = plate
            zone.plate_status = PlateStatus.CONFIRMED

    def set_multi_zone(self, zone_name_a, zone_name_b, plate):
        for zn in [zone_name_a, zone_name_b]:
            zone = self.zones.get(zn)
            if zone:
                zone.status       = ZoneStatus.OCCUPIED
                zone.plate        = plate
                zone.plate_status = PlateStatus.PENDING
                zone.park_status  = ParkStatus.MULTI_ZONE
                zone.entry_time   = time.time()
                zone.no_car_count = 0
                zone.linked_zone  = (
                    zone_name_b if zn == zone_name_a else zone_name_a
                )

    def needs_recheck(self, zone_name: str) -> bool:
        zone = self.zones.get(zone_name)
        if zone is None or zone.status != ZoneStatus.OCCUPIED:
            return False

        # 번호판 확정되면 출차 전까지 재인식 안 함
        if zone.plate_status == PlateStatus.CONFIRMED:
            return False

        # 통로구역은 역추적으로 번호판 부여되므로 재OCR 안 함
        if zone_name in AISLE_ZONES:
            return False

        # NULL/UNREADABLE → 90초마다 재시도
        if (zone.plate_status in (PlateStatus.UNREADABLE, PlateStatus.NULL) and
                (time.time() - zone.last_recheck_time) < RECHECK_INTERVAL_SEC * 3):
            return False
        return (time.time() - zone.last_recheck_time) >= RECHECK_INTERVAL_SEC

    def mark_rechecked(self, zone_name: str):
        zone = self.zones.get(zone_name)
        if zone:
            zone.last_recheck_time     = time.time()
            zone.double_park_suspected = False

    def _reset_zone(self, zone: ZoneState,
                    zone_crop: np.ndarray | None = None):
        zone.status                = ZoneStatus.EMPTY
        zone.plate                 = None
        zone.plate_status          = PlateStatus.PENDING
        zone.park_status           = ParkStatus.NORMAL
        zone.last_foot             = (0, 0)
        zone.still_since           = 0.0
        zone.is_still              = False
        zone.timeout_start         = 0.0
        zone.double_park_suspected = False
        zone.linked_zone           = None
        zone.entry_time            = 0.0
        zone.no_car_count          = 0

        if zone_crop is not None:
            zone.empty_snap           = zone_crop.copy()
            zone.last_mean_brightness = float(
                cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY).mean()
            )
            zone.last_lighting_check  = time.time()

    def to_dict(self) -> dict:
        result = {}
        for name, z in self.zones.items():
            if z.status == ZoneStatus.EMPTY:
                continue
            result[name] = {
                "status":       z.status.value,
                "plate":        z.plate,
                "plate_status": z.plate_status.value,
                "park_status":  z.park_status.value,
                "linked_zone":  z.linked_zone,
                "entry_time":   z.entry_time,
            }
        return result

    def from_dict(self, data: dict):
        for name, info in data.items():
            zone = self.zones.get(name)
            if zone is None:
                continue
            try:
                zone.status       = ZoneStatus(info["status"])
                zone.plate        = info.get("plate")
                zone.plate_status = PlateStatus(
                    info.get("plate_status", "pending")
                )
                zone.park_status  = ParkStatus(
                    info.get("park_status", "normal")
                )
                zone.linked_zone  = info.get("linked_zone")
                zone.entry_time   = info.get("entry_time", 0.0)
                zone.no_car_count = 0
                if zone.status == ZoneStatus.TIMEOUT:
                    zone.status = ZoneStatus.OCCUPIED
            except Exception:
                pass

    def get_all_status(self) -> dict:
        return {
            name: {
                "status":       z.status.value,
                "plate":        z.plate,
                "plate_status": z.plate_status.value,
                "park_status":  z.park_status.value,
                "double_park":  z.double_park_suspected,
                "linked_zone":  z.linked_zone,
                "entry_time":   z.entry_time,
                "has_snap":     z.empty_snap is not None,
            }
            for name, z in self.zones.items()
        }