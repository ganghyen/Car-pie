# ============================================================
# [Phase 3] 구역 상태 머신
#
# 픽셀 비교 로직 수정:
#   기존: 차 있을 때 스냅샷 → 변화 크면 차 있음 (잘못된 로직)
#   수정: 빈 구역일 때 스냅샷 → 변화 크면 차 있음 (올바른 로직)
#
#   빈 구역 스냅샷 저장 시점:
#     1) 매핑 직후 구역 생성 시
#     2) 차량 출차 확정 후
#
#   판단 방식:
#     현재 이미지 vs 빈 구역 스냅샷
#     픽셀 변화 큼  → 차가 들어옴 (차 있음)
#     픽셀 변화 작음 → 비어있음  (차 없음)
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
    PIXEL_DIFF_THRESHOLD,
    PIXEL_CHECK_OCCUPIED,
    PIXEL_LIGHTING_CHANGE_THRESHOLD,
    PIXEL_LIGHTING_UPDATE_INTERVAL,
)


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

    # ★ 빈 구역일 때 찍은 스냅샷 (차 없는 상태 기준)
    empty_snap:           np.ndarray | None = field(default=None, repr=False)

    # 조명 변화 감지용
    last_lighting_check:  float = 0.0
    last_mean_brightness: float = -1.0


class ParkingStateMachine:
    def __init__(self, zone_names: list[str]):
        self.zones: dict[str, ZoneState] = {
            name: ZoneState(name=name) for name in zone_names
        }

    # ── 빈 구역 스냅샷 저장 ───────────────────────────────
    def save_empty_snap(self, zone_name: str, zone_crop: np.ndarray):
        """
        ★ 핵심: 빈 구역일 때 스냅샷 저장
        저장 시점:
          1) 매핑 직후 main.py 에서 호출
          2) 출차 확정 후 자동 호출 (_reset_zone)

        이 이미지를 기준으로 픽셀 비교:
          변화 큼  → 차가 있음
          변화 작음 → 비어있음
        """
        zone = self.zones.get(zone_name)
        if zone:
            zone.empty_snap           = zone_crop.copy()
            zone.last_mean_brightness = float(
                cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY).mean()
            )
            zone.last_lighting_check  = time.time()
            print(f"[PixelCheck] {zone_name} 빈 구역 스냅샷 저장")

    # ── 메인 업데이트 ──────────────────────────────────────
    def update(self, zone_name: str,
               virtual_foot: tuple | None,
               all_cars_in_zone: list[dict],
               plate_visible: bool = True,
               zone_crop: np.ndarray | None = None) -> dict | None:

        zone = self.zones.get(zone_name)
        if zone is None:
            return None

        now       = time.time()
        is_aisle  = zone_name.startswith(AISLE_ZONE_PREFIX)
        still_req = AISLE_STILL_SECONDS if is_aisle else STILL_SECONDS_REQUIRED

        if zone.status == ZoneStatus.OCCUPIED:
            zone.double_park_suspected = (
                len(all_cars_in_zone) >= 2 or not plate_visible
            )

        # 조명 변화 감지 → 빈 스냅샷 업데이트
        # EMPTY 상태일 때만 업데이트 (차 없을 때 기준이어야 하므로)
        if (zone.status == ZoneStatus.EMPTY
                and zone_crop is not None
                and zone.empty_snap is not None):
            self._update_snap_if_lighting_changed(zone, zone_crop, now)

        # YOLO 탐지 여부
        yolo_found    = virtual_foot is not None

        # ★ 픽셀 비교: YOLO 실패 시 빈 스냅샷과 비교
        # 변화가 크면 차가 있는 것
        pixel_has_car = False
        if not yolo_found and zone.status == ZoneStatus.OCCUPIED:
            pixel_has_car = self._pixel_check(zone, zone_crop)

        car_present = yolo_found or pixel_has_car

        # ── EMPTY ──────────────────────────────────────────
        if zone.status == ZoneStatus.EMPTY:
            if yolo_found:
                return self._handle_entry(zone, virtual_foot, now, still_req)

        # ── OCCUPIED ───────────────────────────────────────
        elif zone.status == ZoneStatus.OCCUPIED:
            if car_present:
                zone.timeout_start = 0.0
                return None
            else:
                zone.status        = ZoneStatus.TIMEOUT
                zone.timeout_start = now
                return None

        # ── TIMEOUT ────────────────────────────────────────
        elif zone.status == ZoneStatus.TIMEOUT:
            if car_present:
                zone.status        = ZoneStatus.OCCUPIED
                zone.timeout_start = 0.0
                return None
            else:
                if now - zone.timeout_start >= EXIT_TIMEOUT_SECONDS:
                    old_plate        = zone.plate
                    old_plate_status = zone.plate_status
                    old_entry_time   = zone.entry_time
                    old_status       = zone.park_status
                    old_linked       = zone.linked_zone
                    self._reset_zone(zone, zone_crop)  # 출차 후 빈 스냅샷 갱신
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

    # ── 픽셀 비교 ─────────────────────────────────────────
    def _pixel_check(self, zone: ZoneState,
                     zone_crop: np.ndarray | None) -> bool:
        """
        ★ 빈 구역 스냅샷과 현재 이미지 비교

        차가 있으면 → 픽셀 변화 큼 → True (차 있음)
        차가 없으면 → 픽셀 변화 작음 → False (비어있음)

        이중주차로 원래 차가 가려져도:
        B차가 위에서 찍히므로 빈 구역 스냅샷과 다름 → True (차 있음) ✓
        """
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

            # 변화가 크면 차가 있는 것
            return changed_ratio > PIXEL_CHECK_OCCUPIED

        except Exception:
            return False

    # ── 조명 변화 감지 → 빈 스냅샷 업데이트 ─────────────
    def _update_snap_if_lighting_changed(self, zone, zone_crop, now):
        """
        EMPTY 상태에서만 호출됨 (차 없을 때 기준 유지)
        조명이 크게 바뀌면 빈 구역 스냅샷을 현재 이미지로 교체
        """
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
                zone.empty_snap           = zone_crop.copy()
                zone.last_mean_brightness = mean_bright
                print(f"[PixelCheck] {zone.name} 조명 변화 감지 "
                      f"→ 빈 스냅샷 업데이트")
            else:
                zone.last_mean_brightness = mean_bright
        except Exception:
            pass

    # ── 입차 처리 ──────────────────────────────────────────
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

    # ── 번호판 설정 ────────────────────────────────────────
    def set_plate(self, zone_name: str, plate: str | None):
        from ocr.reader import PLATE_UNREADABLE
        zone = self.zones.get(zone_name)
        if zone is None or zone.status != ZoneStatus.OCCUPIED:
            return
        if plate == PLATE_UNREADABLE:
            zone.plate        = None
            zone.plate_status = PlateStatus.UNREADABLE
        elif plate is None:
        # ★ None 이면 기존 번호판 덮어쓰지 않음
        # FAIL 로 None 반환돼도 이전 번호판 유지
            zone.plate_status = PlateStatus.NULL
        else:
            zone.plate        = plate
            zone.plate_status = PlateStatus.CONFIRMED

    # ── 2칸 주차 ──────────────────────────────────────────
    def set_multi_zone(self, zone_name_a, zone_name_b, plate):
        for zn in [zone_name_a, zone_name_b]:
            zone = self.zones.get(zn)
            if zone:
                zone.status       = ZoneStatus.OCCUPIED
                zone.plate        = plate
                zone.plate_status = PlateStatus.PENDING
                zone.park_status  = ParkStatus.MULTI_ZONE
                zone.entry_time   = time.time()
                zone.linked_zone  = (
                    zone_name_b if zn == zone_name_a else zone_name_a
                )

    # ── 재검증 ────────────────────────────────────────────
    def needs_recheck(self, zone_name: str) -> bool:
        zone = self.zones.get(zone_name)
        if zone is None or zone.status != ZoneStatus.OCCUPIED:
            return False
        if zone.plate_status == PlateStatus.UNREADABLE:
            return False
        now      = time.time()
        periodic = (now - zone.last_recheck_time) >= RECHECK_INTERVAL_SEC
        return periodic or zone.double_park_suspected

    def mark_rechecked(self, zone_name: str):
        zone = self.zones.get(zone_name)
        if zone:
            zone.last_recheck_time     = time.time()
            zone.double_park_suspected = False

    # ── 리셋 (출차 후) ────────────────────────────────────
    def _reset_zone(self, zone: ZoneState,
                    zone_crop: np.ndarray | None = None):
        """
        출차 확정 후 구역 리셋.
        zone_crop 이 있으면 현재 이미지를 새 빈 스냅샷으로 저장.
        """
        # ★ 출차 후 현재 빈 구역 이미지를 스냅샷으로 저장
        if zone_crop is not None:
            zone.empty_snap           = zone_crop.copy()
            zone.last_mean_brightness = float(
                cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY).mean()
            )
            zone.last_lighting_check  = time.time()
            print(f"[PixelCheck] {zone.name} 출차 후 빈 스냅샷 갱신")

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

    # ── 직렬화 (전원 차단 복구용) ─────────────────────────
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
                if zone.status == ZoneStatus.TIMEOUT:
                    zone.status = ZoneStatus.OCCUPIED
                print(f"[StateRestore] {name}: "
                      f"{zone.status.value} / {zone.plate}")
            except Exception as e:
                print(f"[StateRestore] {name} failed: {e}")

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