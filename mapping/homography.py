# ============================================================
# [Phase 1] 호모그래피 좌표 변환 + 카메라 자동 보정
#
# 자동 보정 로직:
#   1) 매 N초마다 아루코 마커 위치 재확인
#   2) 흔들림 감지 시 마커가 보이면 호모그래피 자동 재계산
#   3) 마커가 안 보일 정도로 심하면 경고 + 재매핑 요청
#
# 보정 수준:
#   drift < SHAKE_THRESHOLD       → 정상, 보정 불필요
#   SHAKE_THRESHOLD <= drift < 50 → 자동 보정 실행
#   drift >= 50 또는 마커 소실    → 경고 + 재매핑 권장
# ============================================================

import cv2
import numpy as np
import json
import time
from config.settings import (
    ROI_COORDS_PATH,
    VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT,
    ARUCO_DICT,
    CAMERA_SHAKE_THRESHOLD,
)

ARUCO_DICT_MAP = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

# 자동 보정 불가 수준 (이 이상이면 재매핑 권장)
SHAKE_CRITICAL = 50   # 픽셀


class HomographyTransformer:
    def __init__(self):
        self.matrix      = None
        self.matrix_inv  = None
        self.zones       = {}
        self.map_w       = VIRTUAL_MAP_WIDTH
        self.map_h       = VIRTUAL_MAP_HEIGHT

        # 흔들림 감지용 기준점
        self._ref_marker_centers: dict[int, np.ndarray] | None = None

        # 상태 플래그
        self.camera_shaking   = False   # 흔들림 감지됨
        self.need_remap       = False   # 재매핑 필요 (심한 흔들림)
        self.last_auto_fix    = 0.0     # 마지막 자동 보정 시각
        self.auto_fix_count   = 0       # 누적 자동 보정 횟수

    # ── 로드 ──────────────────────────────────────────────
    def load(self, path: str = ROI_COORDS_PATH) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            matrix_list = data.get("homography_matrix")
            if matrix_list is None:
                return False

            self.matrix     = np.array(matrix_list, dtype=np.float64)
            self.matrix_inv = np.linalg.inv(self.matrix)
            self.zones      = data.get("zones", {})
            print(f"[Homography] Loaded | zones: {len(self.zones)}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"[Homography] Load error: {e}")
            return False

    def is_ready(self) -> bool:
        return self.matrix is not None

    # ── 좌표 변환 ─────────────────────────────────────────
    def camera_to_virtual(self, cam_pt: tuple) -> tuple:
        if self.matrix is None:
            return cam_pt
        pt = np.array([[[float(cam_pt[0]), float(cam_pt[1])]]],
                      dtype=np.float64)
        t  = cv2.perspectiveTransform(pt, self.matrix)
        return (float(t[0][0][0]), float(t[0][0][1]))

    def virtual_to_camera(self, virt_pt: tuple) -> tuple:
        if self.matrix_inv is None:
            return virt_pt
        pt = np.array([[[float(virt_pt[0]), float(virt_pt[1])]]],
                      dtype=np.float64)
        t  = cv2.perspectiveTransform(pt, self.matrix_inv)
        return (float(t[0][0][0]), float(t[0][0][1]))

    @staticmethod
    def bbox_foot(x1, y1, x2, y2) -> tuple:
        return ((x1 + x2) // 2, y2)

    # ── 흔들림 감지 + 자동 보정 ───────────────────────────
    def check_and_auto_correct(self, frame: np.ndarray) -> str:
        """
        아루코 마커 위치를 확인하고 흔들림에 따라 처리합니다.

        Returns:
            "ok"          → 정상 (흔들림 없음)
            "corrected"   → 자동 보정 완료
            "warning"     → 심한 흔들림, 재매핑 권장
            "marker_lost" → 마커 소실, 재매핑 필요
        """
        centers = self._detect_markers(frame)

        # 마커 4개 인식 안 됨
        if len(centers) < 4:
            if self.camera_shaking:
                # 흔들린 상태에서 마커도 소실 → 심각
                self.need_remap = True
                print("[Homography] Markers lost after shake! Re-mapping needed.")
                return "marker_lost"
            return "ok"

        # 최초 기준점 설정
        if self._ref_marker_centers is None:
            self._ref_marker_centers = centers.copy()
            return "ok"

        # 기준점 대비 최대 이동량 계산
        max_drift = 0.0
        for mid, ref_pos in self._ref_marker_centers.items():
            if mid not in centers:
                continue
            drift     = np.linalg.norm(centers[mid] - ref_pos)
            max_drift = max(max_drift, drift)

        # ── 정상 범위 ─────────────────────────────────────
        if max_drift < CAMERA_SHAKE_THRESHOLD:
            if self.camera_shaking:
                # 흔들렸다가 복귀
                self.camera_shaking = False
                self.need_remap     = False
                print("[Homography] Camera stabilized")
            return "ok"

        # ── 자동 보정 가능 범위 ───────────────────────────
        if max_drift < SHAKE_CRITICAL:
            self.camera_shaking = True
            self.need_remap     = False

            # 현재 마커 위치로 호모그래피 재계산
            success = self._recompute_homography(centers)
            if success:
                self.last_auto_fix = time.time()
                self.auto_fix_count += 1
                print(f"[Homography] Auto-corrected "
                      f"(drift: {max_drift:.1f}px, "
                      f"count: {self.auto_fix_count})")
                return "corrected"
            return "warning"

        # ── 심한 흔들림 → 재매핑 권장 ────────────────────
        self.camera_shaking = True
        self.need_remap     = True
        print(f"[Homography] Critical shake! "
              f"drift: {max_drift:.1f}px -> Re-mapping needed")
        return "warning"

    # ── 호모그래피 재계산 ─────────────────────────────────
    def _recompute_homography(self,
                               centers: dict[int, np.ndarray]) -> bool:
        """
        현재 마커 위치로 호모그래피를 재계산합니다.
        성공하면 self.matrix 를 업데이트하고 True 반환.
        """
        try:
            src = np.float32([
                centers[0], centers[1],
                centers[2], centers[3],
            ])
            dst = np.float32([
                [0,                 0                 ],
                [VIRTUAL_MAP_WIDTH, 0                 ],
                [VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT],
                [0,                 VIRTUAL_MAP_HEIGHT],
            ])

            H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                return False

            self.matrix     = H
            self.matrix_inv = np.linalg.inv(H)

            # 기준점도 현재 위치로 업데이트
            self._ref_marker_centers = centers.copy()
            return True

        except Exception as e:
            print(f"[Homography] Recompute failed: {e}")
            return False

    # ── 아루코 마커 감지 ──────────────────────────────────
    def _detect_markers(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        """프레임에서 아루코 마커 중심 좌표를 반환합니다."""
        adict    = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT_MAP.get(ARUCO_DICT, cv2.aruco.DICT_4X4_50)
        )
        params   = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(adict, params)
        corners, ids, _ = detector.detectMarkers(frame)

        centers = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid in [0, 1, 2, 3]:
                    centers[mid] = corners[i][0].mean(axis=0)
        return centers

    # ── 기준점 리셋 ───────────────────────────────────────
    def reset_shake_reference(self, frame: np.ndarray):
        """
        재매핑 후 호출. 흔들림 기준점을 초기화하고
        현재 마커 위치를 새 기준으로 설정합니다.
        """
        self._ref_marker_centers = None
        self.camera_shaking      = False
        self.need_remap          = False
        self.auto_fix_count      = 0

        centers = self._detect_markers(frame)
        if len(centers) >= 4:
            self._ref_marker_centers = centers.copy()
            print("[Homography] Shake reference reset with current markers")
        else:
            print("[Homography] Shake reference reset (markers not found)")