# ============================================================
# [통신] FastAPI 서버로 이벤트 전송
# ⚠️ FastAPI /api/event 엔드포인트로 전송
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
        이벤트를 FastAPI 서버로 전송
        event["type"] 에 따라 FastAPI로 전달
        """
        payload = self._build_payload(event)
        if payload is None:
            return

        # 미전송 큐 먼저 처리
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
                logger.warning(
                    f"[Sender] 서버 응답 오류: {response.status_code}"
                )
                self._enqueue(payload)

        except Exception as e:
            logger.error(f"[Sender] 전송 실패: {e}")
            self._enqueue(payload)

    def _build_payload(self, event: dict) -> dict | None:
        """
        파이 내부 이벤트 → FastAPI 전송 형식으로 변환

        FastAPI /api/event 형식:
        {
            "event":       "entry" / "exit" / "update",
            "zone":        "A-1",
            "plate":       "12가1234" or null,
            "park_type":   "normal" / "multi_zone" / "double_park",
            "linked_zone": "A-2" or null,
            "entry_time":  "2026-05-19 10:00:00",
            "exit_time":   "2026-05-19 11:00:00",
        }
        """
        event_type = event.get("type")
        zone       = event.get("zone")
        timestamp  = event.get("timestamp", 0)
        dt_str     = datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        ) if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 입차 ──────────────────────────────────────────
        if event_type == "entry":
            return {
                "event":       "entry",
                "zone":        zone,
                "plate":       event.get("plate"),
                "park_type":   event.get("park_status", "normal"),
                "linked_zone": event.get("linked_zone"),
                "entry_time":  dt_str,
            }

        # ── 출차 ──────────────────────────────────────────
        elif event_type == "exit":
            return {
                "event":     "exit",
                "zone":      zone,
                "exit_time": dt_str,
            }

        # ── 번호판 업데이트 ───────────────────────────────
        elif event_type in ("plate_update", "plate_changed"):
            return {
                "event": "update",
                "zone":  zone,
                "plate": event.get("plate"),
            }

        else:
            logger.warning(f"[Sender] 알 수 없는 이벤트: {event_type}")
            return None

    # ── 미전송 큐 관리 ────────────────────────────────────
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