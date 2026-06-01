# ============================================================
# [OCR] 번호판 이미지 전처리 (4가지 버전 생성)
# 라즈베리파이 2GB 최적화
# ============================================================

import cv2
import numpy as np
from config.settings import PLATE_UPSCALE, PLATE_PADDING


class PlateEnhancer:

    def crop_plate_region(self, frame: np.ndarray,
                          bbox: dict,
                          plate_bbox: dict | None = None) -> np.ndarray | None:
        """
        번호판 영역을 원본 프레임에서 crop.
        plate_bbox가 있으면 번호판 좌표 직접 사용,
        없으면 차량 bbox 하단 45%를 번호판 추정 영역으로 사용.
        """
        try:
            ih, iw = frame.shape[:2]

            if plate_bbox:
                # YOLO가 번호판을 직접 탐지한 경우
                x1 = max(0, plate_bbox["x1"] - PLATE_PADDING)
                y1 = max(0, plate_bbox["y1"] - PLATE_PADDING)
                x2 = min(iw, plate_bbox["x2"] + PLATE_PADDING)
                y2 = min(ih, plate_bbox["y2"] + PLATE_PADDING)
            else:
                # 번호판 미탐지 시 차량 bbox 하단 영역 사용
                bx1, by1 = bbox["x1"], bbox["y1"]
                bx2, by2 = bbox["x2"], bbox["y2"]
                bh       = by2 - by1
                x1 = max(0, bx1)
                y1 = max(0, by1 + int(bh * 0.55))  # 하단 45% 영역
                x2 = min(iw, bx2)
                y2 = min(ih, by2)

            # 유효하지 않은 crop 범위 체크
            if x2 <= x1 or y2 <= y1:
                return None

            return frame[y1:y2, x1:x2].copy()

        except Exception:
            return None

    def estimate_occlusion_ratio(self, plate_img: np.ndarray) -> float:
        """
        번호판 이미지에서 어두운 픽셀 비율로 가림 정도 추정.
        비율이 높을수록 번호판이 가려진 것으로 판단.
        """
        try:
            gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            # 픽셀값 40 미만을 어두운(가려진) 픽셀로 판단
            dark  = np.sum(gray < 40)
            return dark / gray.size
        except Exception:
            return 0.0

    def generate_variants(self, plate_img: np.ndarray) -> list:
        """
        OCR 인식률 향상을 위해 4가지 전처리 버전 생성.
        여러 버전을 시도해서 가장 좋은 결과를 선택.
        """
        variants = []

        # ── 버전 1: 업스케일 + 언샤프 마스킹 ───────────────
        try:
            h, w  = plate_img.shape[:2]
            # PLATE_UPSCALE 배율로 확대 (기본 3배)
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)
            # 언샤프 마스킹으로 엣지 강화
            sharp = self._unsharp_mask(up)
            variants.append(sharp)
        except Exception:
            variants.append(plate_img)

        # ── 버전 2: 적응형 이진화 ──────────────────────────
        try:
            h, w  = plate_img.shape[:2]
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)
            gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            # 가우시안 적응형 이진화 (조명 불균일 환경에 강함)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        # ── 버전 3: Otsu 이진화 ────────────────────────────
        try:
            h, w  = plate_img.shape[:2]
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)
            gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            # Otsu 방법으로 자동 임계값 결정
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        # ── 버전 4: 언샤프 강화 + 노이즈 제거 + 엣지 강화 ──
        try:
            h, w  = plate_img.shape[:2]
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)
            # 가우시안 블러로 흐린 버전 생성
            gaussian = cv2.GaussianBlur(up, (0, 0), 3)
            # 원본 - 흐린 버전 = 엣지 강화
            sharp = cv2.addWeighted(up, 2.5, gaussian, -1.5, 0)
            # 비지역 평균 필터로 노이즈 제거
            denoised = cv2.fastNlMeansDenoisingColored(
                sharp, None, 10, 10, 7, 21
            )
            # 샤프닝 커널 적용
            kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
            result = cv2.filter2D(denoised, -1, kernel)
            variants.append(result)
        except Exception:
            pass

        return variants if variants else [plate_img]

    @staticmethod
    def _unsharp_mask(img: np.ndarray,
                      kernel_size: int = 5,
                      strength: float = 1.5) -> np.ndarray:
        """언샤프 마스킹: 원본 - 블러 = 엣지, 원본 + 엣지 = 선명한 이미지."""
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        # 원본에 strength 배만큼 엣지를 더해서 선명도 향상
        return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)