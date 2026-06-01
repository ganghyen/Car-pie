# ============================================================
# [통신] FastAPI 서버로 이벤트 전송
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
        # 미전송 큐를 파일에서 복구
        self._pending = self._load_queue()
        logger.info(f"[Sender] 초기화 | 서버: {SERVER_URL}")
        # 복구된 미전송 건수 출력
        if self._pending:
            logger.info(f"[Sender] 미전송 {len(self._pending)}건 복구")

    def send(self, event: dict):
        """
        이벤트를 FastAPI 서버로 전송
        전송 실패 시 로컬 큐에 저장해두고 다음 전송 시 재시도
        """
        # 이벤트를 FastAPI 전송 형식으로 변환
        payload = self._build_payload(event)
        # 변환 실패 (알 수 없는 이벤트 타입 등) 시 전송 중단
        if payload is None:
            return

        # 미전송 큐가 남아있으면 현재 이벤트보다 먼저 재전송 시도
        if self._pending:
            self._flush_pending()

        try:
            # FastAPI 서버로 POST 요청
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
                # 서버가 200 이외 응답 시 큐에 저장
                logger.warning(
                    f"[Sender] 서버 응답 오류: {response.status_code}"
                )
                self._enqueue(payload)

        except Exception as e:
            # 네트워크 오류 등 예외 발생 시 큐에 저장
            logger.error(f"[Sender] 전송 실패: {e}")
            self._enqueue(payload)

    def _build_payload(self, event: dict) -> dict | None:
        """
        파이 내부 이벤트 딕셔너리를 FastAPI 전송 형식으로 변환

        FastAPI /api/event 가 받는 형식:
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
        # timestamp가 있으면 날짜 문자열로 변환, 없으면 현재 시각 사용
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
            # 알 수 없는 이벤트 타입은 경고 후 None 반환
            logger.warning(f"[Sender] 알 수 없는 이벤트: {event_type}")
            return None

    def _enqueue(self, payload: dict):
        # 전송 실패한 payload를 미전송 큐에 추가하고 파일에 저장
        self._pending.append(payload)
        self._save_queue()
        logger.warning(f"[Sender] 큐 저장 ({len(self._pending)}건)")

    def _flush_pending(self):
        # 미전송 큐에 쌓인 이벤트를 순서대로 재전송 시도
        success = []
        for payload in self._pending:
            try:
                r = requests.post(
                    SERVER_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    # 성공한 항목은 제거 목록에 추가
                    success.append(payload)
            except Exception:
                # 재전송도 실패하면 중단 (다음 전송 때 다시 시도)
                break
        # 성공한 항목만 큐에서 제거
        for p in success:
            self._pending.remove(p)
        if success:
            self._save_queue()
            logger.info(f"[Sender] 미전송 {len(success)}건 재전송 완료")

    def _save_queue(self):
        # 미전송 큐를 JSON 파일에 저장
        try:
            os.makedirs(os.path.dirname(QUEUE_FILE_PATH), exist_ok=True)
            with open(QUEUE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._pending, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Sender] 큐 저장 실패: {e}")

    def _load_queue(self) -> list:
        # 큐 파일이 없으면 빈 리스트 반환
        if not os.path.exists(QUEUE_FILE_PATH):
            return []
        try:
            with open(QUEUE_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []