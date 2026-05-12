# ============================================================
# [Phase 4] 오프라인 데이터 큐잉
# 서버 전송 실패 시 로컬에 임시 저장하고 복구 시 재전송합니다.
# ============================================================

import json
import os
import time
from config.settings import QUEUE_FILE_PATH


class EventQueue:
    """
    네트워크 단절 시 입/출차 이벤트를 로컬 파일에 저장합니다.
    서버 연결 복구 시 모아둔 이벤트를 순서대로 재전송합니다.
    """

    def __init__(self):
        self._queue: list[dict] = []
        self._load()

    def push(self, event: dict):
        """이벤트를 큐에 추가하고 파일에 저장합니다."""
        event["queued_at"] = time.time()
        self._queue.append(event)
        self._save()

    def pop_all(self) -> list[dict]:
        """큐의 모든 이벤트를 반환하고 비웁니다."""
        events = self._queue.copy()
        self._queue.clear()
        self._save()
        return events

    def size(self) -> int:
        return len(self._queue)

    def _save(self):
        try:
            os.makedirs(os.path.dirname(QUEUE_FILE_PATH), exist_ok=True)
            with open(QUEUE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._queue, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Queue] 저장 오류: {e}")

    def _load(self):
        try:
            if os.path.exists(QUEUE_FILE_PATH):
                with open(QUEUE_FILE_PATH, "r", encoding="utf-8") as f:
                    self._queue = json.load(f)
                if self._queue:
                    print(f"[Queue] 미전송 이벤트 {len(self._queue)}개 복구됨")
        except Exception as e:
            print(f"[Queue] 로드 오류: {e}")
            self._queue = []