# ============================================================
# [OCR] 번호판 인식 + 다수결 투표
# 패턴 검증 없음 / 유사도 보정은 FastAPI 서버에서 처리
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


# 터미널 출력용 색상 코드
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


# 번호판 인식 불가 상태를 나타내는 특수 문자열
PLATE_UNREADABLE = "UNREADABLE"


class PlateReader:
    def __init__(self, lang: list = None):
        if lang is None:
            lang = ["ko", "en"]
        print(f"[OCR] Initializing EasyOCR...")
        # GPU 미사용 (라즈베리파이 호환)
        self.reader   = easyocr.Reader(lang, gpu=False, verbose=False)
        self.enhancer = PlateEnhancer()
        print(f"[OCR] Ready")

        # 구역별 연속 실패 횟수 추적
        self._fail_count:       dict[str, int]   = {}
        # 구역별 UNREADABLE 판정 시작 시각 (재시도 쿨다운용)
        self._unreadable_since: dict[str, float] = {}

    def read_once(self, plate_img,
                  min_conf: float = None) -> tuple[str | None, float]:
        """
        전처리 4가지 버전 모두 시도 후 가장 신뢰도 높은 결과 반환.
        """
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
                    # 신뢰도가 더 높은 결과로 교체
                    if cleaned and conf > best_conf:
                        best_conf = conf
                        best_text = cleaned
            except Exception:
                continue

        # 신뢰도 임계값 이상인 경우만 반환
        if best_text and best_conf >= min_conf:
            return best_text, best_conf
        return None, best_conf

    def vote_from_snapshot(self, snapshot_frame,
                           bbox: dict,
                           zone_name: str = "",
                           plate_bbox: dict | None = None) -> str | None:
        """
        스냅샷에서 OCR_SAMPLE_COUNT번 인식 후 다수결로 최종 번호판 결정.
        UNREADABLE 쿨다운 중이면 즉시 UNREADABLE 반환.
        """
        # UNREADABLE 쿨다운 체크
        if zone_name in self._unreadable_since:
            elapsed = time.time() - self._unreadable_since[zone_name]
            if elapsed < OCR_UNREADABLE_RETRY_SEC:
                # 쿨다운 중이면 바로 UNREADABLE 반환
                return PLATE_UNREADABLE
            else:
                # 쿨다운 만료 → 재시도 허용
                del self._unreadable_since[zone_name]
                self._fail_count[zone_name] = 0

        # 번호판 이미지 crop
        plate_img = self.enhancer.crop_plate_region(
            snapshot_frame, bbox, plate_bbox=plate_bbox
        )
        if plate_img is None:
            return self._handle_fail(zone_name)

        # 디버깅용 crop 이미지 저장
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

        # 번호판이 70% 이상 가려있으면 실패 처리
        occlusion = self.enhancer.estimate_occlusion_ratio(plate_img)
        if occlusion > 0.7:
            return self._handle_fail(zone_name)

        votes      = []   # 인식 결과 텍스트 투표 목록
        conf_votes = {}   # 텍스트별 최고 신뢰도

        # OCR_SAMPLE_COUNT번 반복 인식
        for _ in range(OCR_SAMPLE_COUNT):
            text, conf = self.read_once(plate_img)
            if text:
                votes.append(text)
                # 같은 텍스트의 최고 신뢰도 갱신
                if text not in conf_votes or conf > conf_votes[text]:
                    conf_votes[text] = conf

        # 투표 결과 출력
        print(f"\n{C.CYAN}[OCR 투표 결과] {zone_name}{C.RESET}")
        all_counter = Counter(votes)
        for text, count in all_counter.most_common():
            print(f"  {text:15s} {count}표  conf:{conf_votes.get(text,0):.2f}")

        if not votes:
            print(f"{C.RED}[OCR] {zone_name} 인식 결과 없음 → null{C.RESET}")
            return self._handle_fail(zone_name)

        # 득표수 × 신뢰도 점수가 가장 높은 텍스트를 최종 선택
        winner = max(
            all_counter.keys(),
            key=lambda t: all_counter[t] * conf_votes.get(t, 0.5)
        )

        # 성공 시 실패 카운터 초기화
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
        # frame_getter 함수 호출로 프레임 취득 후 OCR 실행
        frame = frame_getter()
        if frame is None:
            return None
        return self.vote_from_snapshot(frame, bbox, zone_name, plate_bbox)

    def vote_async(self, frame_getter, bbox: dict,
                   zone_name: str, callback,
                   plate_bbox: dict | None = None) -> threading.Thread:
        """비동기 OCR: 별도 스레드에서 실행 후 callback으로 결과 전달."""
        def _run():
            plate = self.vote(frame_getter, bbox, zone_name, plate_bbox)
            callback(zone_name, plate)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def recheck(self, frame, bbox: dict,
                zone_name: str,
                prev_plate: str | None) -> str | None:
        """
        기존 번호판과 다른 번호판이 인식될 경우에만 반환 (재인식).
        번호판이 변경되지 않았으면 None 반환.
        """
        plate_img = self.enhancer.crop_plate_region(frame, bbox)
        if plate_img is None:
            return None
        # 50% 이상 가려있으면 재인식 중단
        if self.enhancer.estimate_occlusion_ratio(plate_img) > 0.5:
            return None

        # 신뢰도 임계값을 약간 낮춰서 재인식 시도
        new_plate, conf = self.read_once(
            plate_img, min_conf=OCR_CONF_THRESHOLD - 0.1
        )

        # 이전 번호판과 다를 때만 반환
        if new_plate and new_plate != prev_plate:
            print(f"{C.YELLOW}[OCR] Recheck {zone_name}: "
                  f"{prev_plate} → {new_plate} "
                  f"(conf: {conf:.2f}){C.RESET}")
            return new_plate
        return None

    def _handle_fail(self, zone_name: str) -> str | None:
        """OCR 실패 처리: 연속 실패 횟수 증가, 한도 초과 시 UNREADABLE 처리."""
        self._fail_count[zone_name] = \
            self._fail_count.get(zone_name, 0) + 1
        fail_cnt = self._fail_count[zone_name]

        if fail_cnt >= OCR_FAIL_LIMIT:
            # 연속 실패 한도 초과 → UNREADABLE 쿨다운 시작
            self._unreadable_since[zone_name] = time.time()
            print(f"{C.RED}[OCR] {zone_name} UNREADABLE "
                  f"(연속 {fail_cnt}회 실패){C.RESET}")
            return PLATE_UNREADABLE

        print(f"{C.YELLOW}[OCR] {zone_name} FAIL "
              f"({fail_cnt}/{OCR_FAIL_LIMIT}){C.RESET}")
        return None

    @staticmethod
    def _clean(text: str) -> str | None:
        """
        OCR 결과 텍스트 정제.
        공백 제거, 한글/영문/숫자만 남김, 최소 길이 미만은 None 반환.
        """
        text    = text.replace(' ', '').replace('\u00a0', '')
        cleaned = re.sub(r'[^가-힣A-Z0-9a-z]', '', text.upper())
        return cleaned if len(cleaned) >= 5 else None

    def is_unreadable(self, zone_name: str) -> bool:
        # 해당 구역이 UNREADABLE 쿨다운 중인지 반환
        return zone_name in self._unreadable_since

    def reset_unreadable(self, zone_name: str):
        # UNREADABLE 쿨다운 강제 해제 (외부에서 수동 리셋 시)
        self._unreadable_since.pop(zone_name, None)
        self._fail_count[zone_name] = 0