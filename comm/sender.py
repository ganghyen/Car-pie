# ============================================================
# [통신] FastAPI 서버로 이벤트 전송
# 수정사항:
#   1. image_path, ocr_error 필드 추가
#   2. car_image와 image_path 호환 처리
# ============================================================

import requests
import json
import os
from datetime import datetime
from config.settings import SERVER_URL, REQUEST_TIMEOUT, QUEUE_FILE_PATH
from utils.logger import get_logger

logger = get_logger("sender")


class EventSender:
    def __init__(self):
        self._pending = self._load_queue()
        logger.info(f"[Sender] 초기화 | 서버: {SERVER_URL}")
        if self._pending:
            logger.info(f"[Sender] 미전송 {len(self._pending)}건 복구")

    def send(self, event: dict):
        """
        이벤트를 FastAPI 서버로 전송.
        전송 실패 시 로컬 큐에 저장해두고 다음 전송 시 재시도.
        ✅ 수정: HTTP 200이 아니면 실패로 판단해서 재전송 큐에 저장
        """
        payload = self._build_payload(event)
        if payload is None:
            return

        if self._pending:
            self._flush_pending()

        try:
            response = requests.post(
                SERVER_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                logger.info(
                    f"[Sender] 전송 성공 | "
                    f"event:{payload['event']} zone:{payload['zone']}"
                )
            else:
                # ✅ 수정: 200 이외 응답은 모두 실패로 처리 → 큐에 저장
                logger.warning(
                    f"[Sender] 서버 응답 오류: {response.status_code} "
                    f"→ 재전송 큐 저장"
                )
                self._enqueue(payload)

        except Exception as e:
            logger.error(f"[Sender] 전송 실패: {e}")
            self._enqueue(payload)

    def _build_payload(self, event: dict) -> dict | None:
        """
        파이 내부 이벤트 딕셔너리를 FastAPI 전송 형식으로 변환.
        ✅ 수정: image_path, ocr_error 필드 추가
        car_image와 image_path 둘 다 확인해서 호환 처리
        """
        event_type = event.get("type")
        zone       = event.get("zone")
        timestamp  = event.get("timestamp", 0)
        dt_str     = datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        ) if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 입차 이벤트 ───────────────────────────────────
        if event_type == "entry":
            return {
                "event":       "entry",
                "zone":        zone,
                "plate":       event.get("plate"),
                "park_type":   event.get("park_status", "normal"),
                "linked_zone": event.get("linked_zone"),
                "entry_time":  dt_str,
                # ✅ 추가: image_path 전달
                # car_image와 image_path 둘 다 확인 (기존 큐 데이터 호환)
                "image_path":  event.get("image_path") or event.get("car_image"),
                # ✅ 추가: OCR 인식 불가 여부 전달
                "ocr_error":   event.get("ocr_error", False),
            }

        # ── 출차 이벤트 ───────────────────────────────────
        elif event_type == "exit":
            return {
                "event":     "exit",
                "zone":      zone,
                "exit_time": dt_str,
            }

        # ── 번호판 업데이트 이벤트 ────────────────────────
        elif event_type in ("plate_update", "plate_changed"):
            return {
                "event": "update",
                "zone":  zone,
                "plate": event.get("plate"),
            }

        else:
            logger.warning(f"[Sender] 알 수 없는 이벤트: {event_type}")
            return None

    def _enqueue(self, payload: dict):
        self._pending.append(payload)
        self._save_queue()
        logger.warning(f"[Sender] 큐 저장 ({len(self._pending)}건)")

    def _flush_pending(self):
        success = []
        for payload in self._pending:
            try:
                r = requests.post(
                    SERVER_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    success.append(payload)
            except Exception:
                break
        for p in success:
            self._pending.remove(p)
        if success:
            self._save_queue()
            logger.info(f"[Sender] 미전송 {len(success)}건 재전송 완료")

    def _save_queue(self):
        try:
            os.makedirs(os.path.dirname(QUEUE_FILE_PATH), exist_ok=True)
            with open(QUEUE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._pending, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Sender] 큐 저장 실패: {e}")

    def _load_queue(self) -> list:
        if not os.path.exists(QUEUE_FILE_PATH):
            return []
        try:
            with open(QUEUE_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
