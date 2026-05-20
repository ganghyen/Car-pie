# ============================================================
# [OCR] 번호판 인식 + 투표
# ⚠️ 패턴 검증 제거
# ⚠️ 인식된 결과 중 가장 많이 나온 것 선택
# ⚠️ 유사도 보정은 FastAPI 서버에서 처리
# ============================================================

import re
import time
import threading
import cv2
from collections import Counter
import easyocr
from config.settings import (
    OCR_CONF_THRESHOLD, OCR_SAMPLE_COUNT,
    OCR_SAMPLE_INTERVAL, OCR_MIN_TEXT_LENGTH,
    OCR_FAIL_LIMIT, OCR_UNREADABLE_RETRY_SEC,
)
from ocr.enhancer import PlateEnhancer


class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


PLATE_UNREADABLE = "UNREADABLE"


class PlateReader:
    def __init__(self, lang: list = None):
        if lang is None:
            lang = ["ko", "en"]
        print(f"[OCR] Initializing EasyOCR...")
        self.reader   = easyocr.Reader(lang, gpu=False, verbose=False)
        self.enhancer = PlateEnhancer()
        print(f"[OCR] Ready")

        self._fail_count:       dict[str, int]   = {}
        self._unreadable_since: dict[str, float] = {}

    # ── 단일 인식 ──────────────────────────────────────────
    def read_once(self, plate_img,
                  min_conf: float = None) -> tuple[str | None, float]:
        if min_conf is None:
            min_conf = OCR_CONF_THRESHOLD

        variants  = self.enhancer.generate_variants(plate_img)
        best_text = None
        best_conf = 0.0

        for variant in variants:
            try:
                results = self.reader.readtext(variant)
                for (_, text, conf) in results:
                    cleaned = self._clean(text)
                    if cleaned and conf > best_conf:
                        best_conf = conf
                        best_text = cleaned
            except Exception:
                continue

        if best_text and best_conf >= min_conf:
            return best_text, best_conf
        return None, best_conf

    # ── 스냅샷 기반 투표 인식 ─────────────────────────────
    def vote_from_snapshot(self, snapshot_frame,
                           bbox: dict,
                           zone_name: str = "",
                           plate_bbox: dict | None = None) -> str | None:

        if zone_name in self._unreadable_since:
            elapsed = time.time() - self._unreadable_since[zone_name]
            if elapsed < OCR_UNREADABLE_RETRY_SEC:
                return PLATE_UNREADABLE
            else:
                del self._unreadable_since[zone_name]
                self._fail_count[zone_name] = 0

        plate_img = self.enhancer.crop_plate_region(
            snapshot_frame, bbox, plate_bbox=plate_bbox
        )
        if plate_img is None:
            return self._handle_fail(zone_name)

        # crop 이미지 저장 (확인용)
        try:
            import os
            os.makedirs("data/snapshots", exist_ok=True)
            cv2.imwrite(
                f"data/snapshots/plate_crop_{zone_name}.jpg", plate_img
            )
            print(f"[OCR] crop size: "
                  f"{plate_img.shape[1]}x{plate_img.shape[0]}px")
        except Exception:
            pass

        occlusion = self.enhancer.estimate_occlusion_ratio(plate_img)
        if occlusion > 0.7:
            return self._handle_fail(zone_name)

        votes      = []
        conf_votes = {}

        for _ in range(OCR_SAMPLE_COUNT):
            text, conf = self.read_once(plate_img)
            if text:
                votes.append(text)
                if text not in conf_votes or conf > conf_votes[text]:
                    conf_votes[text] = conf

        # 전체 투표 결과 출력
        print(f"\n{C.CYAN}[OCR 투표 결과] {zone_name}{C.RESET}")
        all_counter = Counter(votes)
        for text, count in all_counter.most_common():
            print(f"  {text:15s} {count}표  conf:{conf_votes.get(text,0):.2f}")

        if not votes:
            print(f"{C.RED}[OCR] {zone_name} 인식 결과 없음 → null{C.RESET}")
            return self._handle_fail(zone_name)

        # ★ 패턴 검증 없이 가장 많이 나온 것 선택
        # (유사도 보정은 FastAPI 서버에서 처리)
        winner = max(
            all_counter.keys(),
            key=lambda t: all_counter[t] * conf_votes.get(t, 0.5)
        )

        self._fail_count[zone_name] = 0

        print(f"{C.BOLD}{C.GREEN}"
              f"[OCR 최종] {zone_name} → {winner} "
              f"(conf: {conf_votes.get(winner,0):.2f}, "
              f"{all_counter.get(winner,0)}/{len(votes)}표)"
              f"{C.RESET}")

        return winner

    def vote(self, frame_getter, bbox: dict,
             zone_name: str = "",
             plate_bbox: dict | None = None) -> str | None:
        frame = frame_getter()
        if frame is None:
            return None
        return self.vote_from_snapshot(frame, bbox, zone_name, plate_bbox)

    def vote_async(self, frame_getter, bbox: dict,
                   zone_name: str, callback,
                   plate_bbox: dict | None = None) -> threading.Thread:
        def _run():
            plate = self.vote(frame_getter, bbox, zone_name, plate_bbox)
            callback(zone_name, plate)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def recheck(self, frame, bbox: dict,
                zone_name: str,
                prev_plate: str | None) -> str | None:
        plate_img = self.enhancer.crop_plate_region(frame, bbox)
        if plate_img is None:
            return None
        if self.enhancer.estimate_occlusion_ratio(plate_img) > 0.5:
            return None

        new_plate, conf = self.read_once(
            plate_img, min_conf=OCR_CONF_THRESHOLD - 0.1
        )

        # ★ 패턴 검증 제거 - 그대로 반환
        if new_plate and new_plate != prev_plate:
            print(f"{C.YELLOW}[OCR] Recheck {zone_name}: "
                  f"{prev_plate} → {new_plate} "
                  f"(conf: {conf:.2f}){C.RESET}")
            return new_plate
        return None

    def _handle_fail(self, zone_name: str) -> str | None:
        self._fail_count[zone_name] = \
            self._fail_count.get(zone_name, 0) + 1
        fail_cnt = self._fail_count[zone_name]

        if fail_cnt >= OCR_FAIL_LIMIT:
            self._unreadable_since[zone_name] = time.time()
            print(f"{C.RED}[OCR] {zone_name} UNREADABLE "
                  f"(연속 {fail_cnt}회 실패){C.RESET}")
            return PLATE_UNREADABLE

        print(f"{C.YELLOW}[OCR] {zone_name} FAIL "
              f"({fail_cnt}/{OCR_FAIL_LIMIT}){C.RESET}")
        return None

    @staticmethod
    def _clean(text: str) -> str | None:
        text    = text.replace(' ', '').replace('\u00a0', '')
        cleaned = re.sub(r'[^가-힣A-Z0-9a-z]', '', text.upper())
        return cleaned if len(cleaned) >= 5 else None

    def is_unreadable(self, zone_name: str) -> bool:
        return zone_name in self._unreadable_since

    def reset_unreadable(self, zone_name: str):
        self._unreadable_since.pop(zone_name, None)
        self._fail_count[zone_name] = 0