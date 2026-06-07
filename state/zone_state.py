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
    PIXEL_DIFF_THRESHOLD,
    PIXEL_CHECK_OCCUPIED,
    PIXEL_LIGHTING_CHANGE_THRESHOLD,
    PIXEL_LIGHTING_UPDATE_INTERVAL,
)


class ZoneStatus(Enum):
    EMPTY    = "empty"     # 빈 구역
    OCCUPIED = "occupied"  # 차량 점유 중
    TIMEOUT  = "timeout"   # 차가 안 보이기 시작 (출차 확인 대기)


class ParkStatus(Enum):
    NORMAL      = "normal"       # 일반 주차
    DOUBLE_PARK = "double_park"  # 이중주차
    MULTI_ZONE  = "multi_zone"   # 2칸 주차
    AISLE_BLOCK = "aisle_block"  # 통로 막음


class PlateStatus(Enum):
    PENDING    = "pending"     # OCR 인식 대기 중
    CONFIRMED  = "confirmed"   # 번호판 확정
    NULL       = "null"        # 번호판 없음 (차는 있음)
    UNREADABLE = "unreadable"  # 번호판 인식 불가


@dataclass
class ZoneState:
    # 구역 이름
    name:             str
    # 현재 구역 상태 (기본값: EMPTY)
    status:           ZoneStatus  = ZoneStatus.EMPTY
    # 현재 점유 차량 번호판
    plate:            str | None  = None
    # 번호판 인식 상태
    plate_status:     PlateStatus = PlateStatus.PENDING
    # 주차 유형
    park_status:      ParkStatus  = ParkStatus.NORMAL

    # 차량 발바닥 마지막 위치 (정지 판정용)
    last_foot:    tuple = field(default_factory=lambda: (0, 0))
    # 정지 시작 시각
    still_since:  float = 0.0
    # 현재 정지 상태 여부
    is_still:     bool  = False

    # TIMEOUT 진입 시각 (출차 타이머 기준)
    timeout_start:     float = 0.0
    # 마지막 번호판 재인식 시각
    last_recheck_time: float = 0.0

    # 이중주차 의심 여부
    double_park_suspected: bool       = False
    # 2칸 주차 시 연결된 구역 이름
    linked_zone:           str | None = None
    # 입차 시각
    entry_time:            float      = 0.0

    # 빈 구역 기준 스냅샷 (픽셀 비교용, 직렬화 제외)
    empty_snap:           np.ndarray | None = field(default=None, repr=False)

    # 조명 변화 감지용 마지막 체크 시각
    last_lighting_check:  float = 0.0
    # 조명 변화 감지용 마지막 평균 밝기값
    last_mean_brightness: float = -1.0


class ParkingStateMachine:
    def __init__(self, zone_names: list[str]):
        # 구역 이름 목록으로 ZoneState 딕셔너리 초기화
        self.zones: dict[str, ZoneState] = {
            name: ZoneState(name=name) for name in zone_names
        }

    def save_empty_snap(self, zone_name: str, zone_crop: np.ndarray):
        """
        빈 구역 스냅샷 저장.
        저장 시점: 1) 매핑 직후  2) 출차 확정 후
        이 이미지 기준으로 픽셀 비교 → 차 유무 판단
        """
        zone = self.zones.get(zone_name)
        if zone:
            # 현재 프레임을 빈 구역 기준 이미지로 저장
            zone.empty_snap           = zone_crop.copy()
            # 현재 평균 밝기 저장 (조명 변화 감지 기준)
            zone.last_mean_brightness = float(
                cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY).mean()
            )
            zone.last_lighting_check  = time.time()
            print(f"[PixelCheck] {zone_name} 빈 구역 스냅샷 저장")

    def update(self, zone_name: str,
               virtual_foot: tuple | None,
               all_cars_in_zone: list[dict],
               plate_visible: bool = True,
               zone_crop: np.ndarray | None = None) -> dict | None:
        """
        매 프레임 호출. 구역 상태를 갱신하고 이벤트 발생 시 딕셔너리 반환.
        반환값: entry/exit 이벤트 딕셔너리 or None
        """
        zone = self.zones.get(zone_name)
        if zone is None:
            return None

        now = time.time()
        # 통로 구역 여부에 따라 정지 판정 시간 선택
        is_aisle  = zone_name.startswith(AISLE_ZONE_PREFIX)
        still_req = AISLE_STILL_SECONDS if is_aisle else STILL_SECONDS_REQUIRED

        # OCCUPIED 상태에서 이중주차 의심 여부 갱신
        if zone.status == ZoneStatus.OCCUPIED:
            zone.double_park_suspected = (
                len(all_cars_in_zone) >= 2 or not plate_visible
            )

        # EMPTY 상태에서만 조명 변화 체크 (차 없을 때 기준 유지)
        if (zone.status == ZoneStatus.EMPTY
                and zone_crop is not None
                and zone.empty_snap is not None):
            self._update_snap_if_lighting_changed(zone, zone_crop, now)

        # YOLO 탐지 여부
        yolo_found = virtual_foot is not None

        # YOLO가 못 찾았을 때 OCCUPIED/TIMEOUT 구역은 픽셀 비교로 보조 판단
        pixel_has_car = False
        if not yolo_found and zone.status in (ZoneStatus.OCCUPIED, ZoneStatus.TIMEOUT):
            pixel_has_car = self._pixel_check(zone, zone_crop)

        # 최종 차량 존재 여부 (YOLO OR 픽셀비교)
        car_present = yolo_found or pixel_has_car

        # ── EMPTY 상태 ─────────────────────────────────────
        if zone.status == ZoneStatus.EMPTY:
            if yolo_found:
                # 차가 감지되면 입차 처리 시도
                return self._handle_entry(zone, virtual_foot, now, still_req)

        # ── OCCUPIED 상태 ──────────────────────────────────
        elif zone.status == ZoneStatus.OCCUPIED:
            if car_present:
                # 차가 계속 있음 → timeout_start 초기화
                zone.timeout_start = 0.0
                return None
            else:
                # 차가 처음 사라짐 → TIMEOUT 진입
                zone.status        = ZoneStatus.TIMEOUT
                zone.timeout_start = now
                print(f"[STATE] {zone_name} OCCUPIED→TIMEOUT plate={zone.plate}")
                return None

        # ── TIMEOUT 상태 ───────────────────────────────────
        elif zone.status == ZoneStatus.TIMEOUT:
            if car_present:
                # 차가 다시 감지됨 → OCCUPIED 복귀
                zone.status        = ZoneStatus.OCCUPIED
                zone.timeout_start = 0.0
                print(f"[STATE] {zone_name} TIMEOUT→OCCUPIED (재감지)")
                return None

            # timeout_start가 0이면 비정상 상태 → 현재 시각으로 보정
            if zone.timeout_start == 0.0:
                zone.timeout_start = now
                print(f"[WARN] {zone_name} timeout_start=0 보정")
                return None

            elapsed = now - zone.timeout_start
            if elapsed >= EXIT_TIMEOUT_SECONDS:
                # 타이머 초과 → 출차 확정
                old_plate        = zone.plate
                old_plate_status = zone.plate_status
                old_entry_time   = zone.entry_time
                old_status       = zone.park_status
                old_linked       = zone.linked_zone
                # 구역 리셋 + 빈 스냅샷 갱신
                self._reset_zone(zone, zone_crop)
                print(f"[EVENT] exit | zone={zone_name} "
                      f"plate={old_plate} elapsed={elapsed:.1f}s")
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
            else:
                # 아직 타이머 중
                print(f"[STATE] {zone_name} TIMEOUT 대기 "
                      f"{elapsed:.1f}/{EXIT_TIMEOUT_SECONDS}s")

        return None

    def _pixel_check(self, zone: ZoneState,
                     zone_crop: np.ndarray | None) -> bool:
        """
        빈 구역 스냅샷과 현재 이미지를 비교해서 차량 유무 판단.
        변화 비율 > PIXEL_CHECK_OCCUPIED 이면 차가 있다고 판단.
        """
        if zone.empty_snap is None or zone_crop is None:
            return False

        try:
            snap = zone.empty_snap
            curr = zone_crop

            # 크기가 다르면 스냅샷 크기로 리사이즈
            if snap.shape != curr.shape:
                curr = cv2.resize(curr, (snap.shape[1], snap.shape[0]))

            # 그레이스케일로 변환 후 픽셀 차이 계산
            snap_gray = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            diff          = cv2.absdiff(snap_gray, curr_gray)
            # 임계값 이상 변화된 픽셀 비율 계산
            changed_ratio = np.sum(diff > PIXEL_DIFF_THRESHOLD) / diff.size

            # 변화 비율이 기준 초과 시 차가 있다고 판단
            return changed_ratio > PIXEL_CHECK_OCCUPIED

        except Exception:
            return False

    def _update_snap_if_lighting_changed(self, zone, zone_crop, now):
        """
        EMPTY 상태에서만 호출.
        조명이 크게 바뀌면 빈 구역 스냅샷을 현재 이미지로 교체.
        """
        # 업데이트 주기가 되지 않았으면 스킵
        if now - zone.last_lighting_check < PIXEL_LIGHTING_UPDATE_INTERVAL:
            return
        zone.last_lighting_check = now

        try:
            gray        = cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY)
            mean_bright = float(gray.mean())

            if zone.last_mean_brightness < 0:
                # 최초 측정 시 기준값만 저장
                zone.last_mean_brightness = mean_bright
                return

            # 밝기 변화가 임계값 초과 시 스냅샷 교체
            if abs(mean_bright - zone.last_mean_brightness) \
                    > PIXEL_LIGHTING_CHANGE_THRESHOLD:
                zone.empty_snap           = zone_crop.copy()
                zone.last_mean_brightness = mean_bright
                print(f"[PixelCheck] {zone.name} 조명 변화 감지 "
                      f"→ 빈 스냅샷 업데이트")
            else:
                # 변화 없으면 밝기값만 갱신
                zone.last_mean_brightness = mean_bright
        except Exception:
            pass

    def _handle_entry(self, zone, foot, now, still_req) -> dict | None:
        """
        EMPTY 상태에서 차량 감지 시 호출.
        차량이 STILL_SECONDS_REQUIRED 초 이상 정지하면 입차 확정.
        """
        # 발바닥 이동량 계산
        dx = abs(foot[0] - zone.last_foot[0])
        dy = abs(foot[1] - zone.last_foot[1])
        zone.last_foot = foot

        if max(dx, dy) < STILL_PIXEL_THRESHOLD:
            # 이동량이 임계값 미만 → 정지 상태
            if not zone.is_still:
                # 정지 시작 시각 기록
                zone.is_still    = True
                zone.still_since = now
            elif now - zone.still_since >= still_req:
                # 정지 시간이 기준 이상 → 입차 확정
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
            # 이동량이 임계값 이상 → 아직 움직이는 중, 정지 초기화
            zone.is_still    = False
            zone.still_since = 0.0
        return None

    def set_plate(self, zone_name: str, plate: str | None):
        """OCR 결과를 구역 상태에 반영. OCCUPIED 상태에서만 동작."""
        from ocr.reader import PLATE_UNREADABLE
        zone = self.zones.get(zone_name)
        if zone is None or zone.status != ZoneStatus.OCCUPIED:
            return

        if plate == PLATE_UNREADABLE:
            # 인식 불가 판정
            zone.plate        = None
            zone.plate_status = PlateStatus.UNREADABLE
        elif plate is None:
            # OCR 실패(None) → 기존 번호판 유지, 상태만 NULL로 변경
            zone.plate_status = PlateStatus.NULL
        else:
            # 정상 인식 → 번호판 저장
            zone.plate        = plate
            zone.plate_status = PlateStatus.CONFIRMED

    def set_multi_zone(self, zone_name_a, zone_name_b, plate):
        """2칸 주차 처리: 두 구역을 동시에 OCCUPIED로 설정하고 서로 linked."""
        for zn in [zone_name_a, zone_name_b]:
            zone = self.zones.get(zn)
            if zone:
                zone.status       = ZoneStatus.OCCUPIED
                zone.plate        = plate
                zone.plate_status = PlateStatus.PENDING
                zone.park_status  = ParkStatus.MULTI_ZONE
                zone.entry_time   = time.time()
                # 연결된 상대 구역 이름 저장
                zone.linked_zone  = (
                    zone_name_b if zn == zone_name_a else zone_name_a
                )

    def needs_recheck(self, zone_name: str) -> bool:
        """번호판 재인식이 필요한지 판단."""
        zone = self.zones.get(zone_name)
        if zone is None or zone.status != ZoneStatus.OCCUPIED:
            return False
        # UNREADABLE 판정된 구역은 재인식 시도 안 함
        if zone.plate_status == PlateStatus.UNREADABLE:
            return False
        now      = time.time()
        # 주기적 재인식 또는 이중주차 의심 시 재인식
        periodic = (now - zone.last_recheck_time) >= RECHECK_INTERVAL_SEC
        return periodic or zone.double_park_suspected

    def mark_rechecked(self, zone_name: str):
        """재인식 완료 처리: 마지막 재인식 시각 갱신, 이중주차 의심 해제."""
        zone = self.zones.get(zone_name)
        if zone:
            zone.last_recheck_time     = time.time()
            zone.double_park_suspected = False

    def _reset_zone(self, zone: ZoneState,
                    zone_crop: np.ndarray | None = None):
        """출차 확정 후 구역 초기화. zone_crop이 있으면 빈 스냅샷도 갱신."""
        if zone_crop is not None:
            # 현재 프레임을 새 빈 구역 기준 이미지로 저장
            zone.empty_snap           = zone_crop.copy()
            zone.last_mean_brightness = float(
                cv2.cvtColor(zone_crop, cv2.COLOR_BGR2GRAY).mean()
            )
            zone.last_lighting_check  = time.time()
            print(f"[PixelCheck] {zone.name} 출차 후 빈 스냅샷 갱신")

        # 모든 상태값 초기화
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

    def to_dict(self) -> dict:
        """전원 차단 복구용 직렬화. EMPTY 구역은 저장 제외."""
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
        """직렬화된 상태 복구. TIMEOUT 상태는 안전하게 OCCUPIED로 복구."""
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
                # TIMEOUT 상태로 저장된 경우 OCCUPIED로 복구 (안전)
                if zone.status == ZoneStatus.TIMEOUT:
                    zone.status = ZoneStatus.OCCUPIED
                print(f"[StateRestore] {name}: "
                      f"{zone.status.value} / {zone.plate}")
            except Exception as e:
                print(f"[StateRestore] {name} failed: {e}")

    def get_all_status(self) -> dict:
        """현재 전체 구역 상태를 딕셔너리로 반환 (시각화용)."""
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