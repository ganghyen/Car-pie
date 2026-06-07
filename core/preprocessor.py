# ============================================================
# [Phase 2] 전처리 + 카메라 이물질 감지
# 라즈베리파이 2GB 최적화: CLAHE를 N프레임마다 1번만 실행
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
CLAHE_INTERVAL = 10


class Preprocessor:
    def __init__(self, clip_limit: float = 2.0,
                 tile_grid: tuple = (8, 8)):
        # CLAHE 객체 생성 (대비 제한값, 타일 크기 설정)
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid
        )
        # CLAHE 적용 프레임 카운터
        self._frame_count   = 0
        # 마지막으로 CLAHE 적용한 결과 캐시 (같은 결과 재사용)
        self._cached_frame  = None

        # 흐림 감지 관련 변수
        self._last_blur_check   = 0.0   # 마지막 흐림 체크 시각
        self._blur_count        = 0     # 연속 흐림 감지 횟수
        self.camera_blurry      = False # 카메라 오염 여부 플래그
        self.current_sharpness  = 999.0 # 현재 선명도 값

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        N프레임마다 1번 CLAHE 적용.
        나머지 프레임은 캐시된 결과 반환.
        라즈베리파이 CPU 부하를 줄이기 위한 최적화.
        """
        self._frame_count += 1

        # CLAHE 적용 주기가 됐거나 처음 실행이면 실제 적용
        if (self._frame_count % CLAHE_INTERVAL == 0
                or self._cached_frame is None):
            # BGR → LAB 색공간 변환 (L채널에만 CLAHE 적용)
            lab         = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b     = cv2.split(lab)
            # L채널(밝기)에만 CLAHE 적용
            l_eq        = self.clahe.apply(l)
            lab_eq      = cv2.merge([l_eq, a, b])
            # 다시 BGR로 변환
            enhanced    = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            # 결과 캐시에 저장
            self._cached_frame = enhanced
            return enhanced
        else:
            # 주기가 아니면 캐시된 이전 결과 반환 (CPU 절약)
            return self._cached_frame

    def check_blur(self, frame: np.ndarray) -> bool:
        """
        라플라시안 분산으로 카메라 선명도를 측정.
        선명도가 임계값 미만이 BLUR_CONFIRM_COUNT회 연속이면 오염으로 확정.
        """
        now = time.time()
        # 체크 주기가 되지 않았으면 이전 결과 반환
        if now - self._last_blur_check < BLUR_CHECK_INTERVAL:
            return self.camera_blurry
        self._last_blur_check = now

        try:
            # 그레이스케일 변환 후 라플라시안 분산 계산
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            self.current_sharpness = sharpness

            if sharpness < BLUR_DETECT_THRESHOLD:
                # 선명도 부족 → 연속 카운터 증가
                self._blur_count += 1
                if self._blur_count >= BLUR_CONFIRM_COUNT:
                    # 연속 N회 이상 흐림이면 오염으로 확정
                    if not self.camera_blurry:
                        print(f"[Preprocessor] Camera blur detected! "
                              f"sharpness: {sharpness:.1f}")
                    self.camera_blurry = True
            else:
                # 선명도 회복 시 오염 해제
                if self.camera_blurry:
                    print(f"[Preprocessor] Camera clear. "
                          f"sharpness: {sharpness:.1f}")
                self._blur_count   = 0
                self.camera_blurry = False

        except Exception as e:
            print(f"[Preprocessor] Blur check error: {e}")

        return self.camera_blurry