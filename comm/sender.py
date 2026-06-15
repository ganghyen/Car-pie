# ============================================================
# [통신] FastAPI 서버로 이벤트 전송 (큐 없음 - 실패 시 버림)
# ============================================================

import requests
from datetime import datetime
from config.settings import (
    SERVER_URL,
    REQUEST_TIMEOUT,
    APARTMENT_NO,
)
from utils.logger import get_logger

logger = get_logger("sender")


class EventSender:
    def __init__(self):
        logger.info(f"[Sender] 초기화 | 서버: {SERVER_URL}")

    def send(self, event: dict):
        payload = self._build_payload(event)
        if payload is None:
            return

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
                    f"[Sender] 서버 응답 오류: {response.status_code} (버림)"
                )

        except Exception as e:
            logger.error(f"[Sender] 전송 실패: {e} (버림)")

    def _build_payload(self, event: dict) -> dict | None:
        event_type = event.get("type")
        zone       = event.get("zone")
        timestamp  = event.get("timestamp", 0)
        dt_str     = datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        ) if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if event_type == "entry_quick":
            return {
                "event":        "entry_quick",
                "zone":         zone,
                "plate":        None,
                "park_type":    event.get("park_status", "normal"),
                "linked_zone":  event.get("linked_zone"),
                "entry_time":   dt_str,
                "apartment_no": event.get("apartment_no") or APARTMENT_NO,
            }

        if event_type == "entry":
            return {
                "event":        "entry",
                "zone":         zone,
                "plate":        event.get("plate"),
                "park_type":    event.get("park_status", "normal"),
                "linked_zone":  event.get("linked_zone"),
                "entry_time":   dt_str,
                "apartment_no": event.get("apartment_no") or APARTMENT_NO,
                "image_base64": event.get("image_base64"),
                "ocr_error":    event.get("ocr_error", False),
            }

        elif event_type == "exit":
            return {
                "event":     "exit",
                "zone":      zone,
                "exit_time": dt_str,
            }

        elif event_type in ("plate_update", "plate_changed"):
            return {
                "event": "update",
                "zone":  zone,
                "plate": event.get("plate"),
            }

        else:
            logger.warning(f"[Sender] 알 수 없는 이벤트: {event_type}")
            return None