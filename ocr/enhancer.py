# ============================================================
# [OCR] 번호판 이미지 전처리
#
# 라즈베리파이 2GB 최적화:
#   1. 업스케일 + 언샤프 마스킹
#   2. 적응형 이진화
#   3. Otsu 이진화
#   4. 언샤프 마스킹 강화 + 노이즈 제거 + 엣지 강화 (AI보정 느낌)
# ============================================================

import cv2
import numpy as np
from config.settings import PLATE_UPSCALE, PLATE_PADDING


class PlateEnhancer:

    # ── 번호판 영역 crop ──────────────────────────────────
    def crop_plate_region(self, frame: np.ndarray,
                          bbox: dict,
                          plate_bbox: dict | None = None) -> np.ndarray | None:
        try:
            ih, iw = frame.shape[:2]

            if plate_bbox:
                x1 = max(0, plate_bbox["x1"] - PLATE_PADDING)
                y1 = max(0, plate_bbox["y1"] - PLATE_PADDING)
                x2 = min(iw, plate_bbox["x2"] + PLATE_PADDING)
                y2 = min(ih, plate_bbox["y2"] + PLATE_PADDING)
            else:
                bx1, by1 = bbox["x1"], bbox["y1"]
                bx2, by2 = bbox["x2"], bbox["y2"]
                bh       = by2 - by1
                x1 = max(0, bx1)
                y1 = max(0, by1 + int(bh * 0.55))
                x2 = min(iw, bx2)
                y2 = min(ih, by2)

            if x2 <= x1 or y2 <= y1:
                return None

            return frame[y1:y2, x1:x2].copy()

        except Exception:
            return None

    # ── 가림 비율 추정 ────────────────────────────────────
    def estimate_occlusion_ratio(self, plate_img: np.ndarray) -> float:
        try:
            gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            dark  = np.sum(gray < 40)
            return dark / gray.size
        except Exception:
            return 0.0

    # ── 전처리 버전 생성 (4가지) ──────────────────────────
    def generate_variants(self, plate_img: np.ndarray) -> list:
        """
        4가지 전처리 버전 생성
          1. 업스케일 + 언샤프 마스킹
          2. 적응형 이진화
          3. Otsu 이진화
          4. 언샤프 마스킹 강화 + 노이즈 제거 + 엣지 강화
        """
        variants = []

        # ── 1. 업스케일 + 언샤프 마스킹 ──────────────────
        try:
            h, w  = plate_img.shape[:2]
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)
            sharp = self._unsharp_mask(up)
            variants.append(sharp)
        except Exception:
            variants.append(plate_img)

        # ── 2. 적응형 이진화 ──────────────────────────────
        try:
            h, w  = plate_img.shape[:2]
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)
            gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        # ── 3. Otsu 이진화 ────────────────────────────────
        try:
            h, w  = plate_img.shape[:2]
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)
            gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        # ── 4. 언샤프 마스킹 강화 + 노이즈 제거 + 엣지 강화
        try:
            h, w  = plate_img.shape[:2]
            up    = cv2.resize(plate_img,
                               (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                               interpolation=cv2.INTER_CUBIC)

            # 가우시안 블러로 흐린 버전 생성
            gaussian = cv2.GaussianBlur(up, (0, 0), 3)

            # 원본에서 흐린 버전 빼서 엣지 강화
            sharp = cv2.addWeighted(up, 2.5, gaussian, -1.5, 0)

            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoisingColored(
                sharp, None, 10, 10, 7, 21
            )

            # 엣지 강화 커널
            kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
            result = cv2.filter2D(denoised, -1, kernel)
            variants.append(result)
        except Exception:
            pass

        return variants if variants else [plate_img]

    # ── 언샤프 마스킹 ─────────────────────────────────────
    @staticmethod
    def _unsharp_mask(img: np.ndarray,
                      kernel_size: int = 5,
                      strength: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)