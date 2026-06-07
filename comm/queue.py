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
        # 메모리 내 이벤트 큐 초기화 후 파일에서 미전송 이벤트 복구
        self._queue: list[dict] = []
        self._load()

    def push(self, event: dict):
        """이벤트를 큐에 추가하고 파일에 저장합니다."""
        # 큐에 넣은 시각을 이벤트에 기록
        event["queued_at"] = time.time()
        self._queue.append(event)
        # 메모리 큐 변경 즉시 파일에 반영 (프로세스 종료 시 유실 방지)
        self._save()

    def pop_all(self) -> list[dict]:
        """큐의 모든 이벤트를 반환하고 비웁니다."""
        # 현재 큐 전체 복사본 반환
        events = self._queue.copy()
        # 메모리 큐 초기화
        self._queue.clear()
        # 빈 상태를 파일에도 반영
        self._save()
        return events

    def size(self) -> int:
        # 현재 큐에 쌓인 이벤트 수 반환
        return len(self._queue)

    def _save(self):
        # 큐 파일이 있는 디렉토리가 없으면 생성
        try:
            os.makedirs(os.path.dirname(QUEUE_FILE_PATH), exist_ok=True)
            # 현재 큐 내용을 JSON 파일로 저장
            with open(QUEUE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._queue, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Queue] 저장 오류: {e}")

    def _load(self):
        # 큐 파일이 존재하면 읽어서 메모리 큐에 복구
        try:
            if os.path.exists(QUEUE_FILE_PATH):
                with open(QUEUE_FILE_PATH, "r", encoding="utf-8") as f:
                    self._queue = json.load(f)
                # 복구된 미전송 이벤트 수 출력
                if self._queue:
                    print(f"[Queue] 미전송 이벤트 {len(self._queue)}개 복구됨")
        except Exception as e:
            print(f"[Queue] 로드 오류: {e}")
            # 로드 실패 시 빈 큐로 초기화
            self._queue = []