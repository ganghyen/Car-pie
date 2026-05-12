# ============================================================
# [Phase 2] 전처리 + 카메라 이물질 감지
#
# 라즈베리파이 2GB 최적화:
#   CLAHE → 매 프레임 X, N프레임마다 1번
#   지하주차장 조명이 일정하면 매 프레임마다 할 필요 없음
#   CLAHE_INTERVAL 로 간격 조정 (기본 5프레임마다 1번)
# ============================================================

import cv2
import numpy as np
import time
from config.settings import (
    BLUR_DETECT_THRESHOLD,
    BLUR_CHECK_INTERVAL,
    BLUR_CONFIRM_COUNT,
)

# CLAHE 적용 간격 (N프레임마다 1번)
# 조명이 일정한 실내: 10~30 권장
# 조명 변화 있는 환경: 3~5 권장
CLAHE_INTERVAL = 10


class Preprocessor:
    def __init__(self, clip_limit: float = 2.0,
                 tile_grid: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid
        )

        # CLAHE 캐시
        self._frame_count   = 0
        self._cached_frame  = None   # 마지막 CLAHE 적용 프레임

        # 흐림 감지
        self._last_blur_check   = 0.0
        self._blur_count        = 0
        self.camera_blurry      = False
        self.current_sharpness  = 999.0

    # ── CLAHE 전처리 ──────────────────────────────────────
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        N프레임마다 1번 CLAHE 적용.
        나머지 프레임은 캐시된 결과 반환.

        지하주차장처럼 조명이 일정한 환경에서
        매 프레임마다 CLAHE 할 필요 없어요.
        라즈베리파이 CPU 부하 줄이는 효과.
        """
        self._frame_count += 1

        if (self._frame_count % CLAHE_INTERVAL == 0
                or self._cached_frame is None):
            # CLAHE 적용
            lab         = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b     = cv2.split(lab)
            l_eq        = self.clahe.apply(l)
            lab_eq      = cv2.merge([l_eq, a, b])
            enhanced    = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            self._cached_frame = enhanced
            return enhanced
        else:
            # 캐시 반환 (CLAHE 생략)
            return self._cached_frame

    # ── 카메라 선명도 체크 ────────────────────────────────
    def check_blur(self, frame: np.ndarray) -> bool:
        now = time.time()
        if now - self._last_blur_check < BLUR_CHECK_INTERVAL:
            return self.camera_blurry

        self._last_blur_check = now

        try:
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            self.current_sharpness = sharpness

            if sharpness < BLUR_DETECT_THRESHOLD:
                self._blur_count += 1
                if self._blur_count >= BLUR_CONFIRM_COUNT:
                    if not self.camera_blurry:
                        print(f"[Preprocessor] Camera blur detected! "
                              f"sharpness: {sharpness:.1f}")
                    self.camera_blurry = True
            else:
                if self.camera_blurry:
                    print(f"[Preprocessor] Camera clear. "
                          f"sharpness: {sharpness:.1f}")
                self._blur_count   = 0
                self.camera_blurry = False

        except Exception as e:
            print(f"[Preprocessor] Blur check error: {e}")

        return self.camera_blurry