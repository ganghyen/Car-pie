# ============================================================
# [Phase 4] 서버/DB 전송
#
# DB 포맷이 바뀌면 _build_payload() 함수만 수정하면 됩니다.
#
# 전송 포맷:
# {
#   "event"       : "entry" | "exit" | "update",
#   "zone"        : "A-1" | "A-1,A-2" | null (구역 미설정 시),
#   "plate"       : "12가3456" | null,
#   "plate_status": "confirmed" | "null" | "unreadable" | "pending",
#   "entry_time"  : "2025-01-15 14:32:10",
#   "exit_time"   : "2025-01-15 15:10:05" | null,
#   "park_status" : "normal"|"double_park"|"multi_zone"|"aisle_block",
#   "car_image"   : "data/snapshots/A-1_20250115_143210.jpg" | null
# }
# ============================================================

import requests
import time
from datetime import datetime
from comm.queue import EventQueue
from config.settings import SERVER_URL, REQUEST_TIMEOUT


# 터미널 색상
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


class EventSender:
    def __init__(self):
        self.queue        = EventQueue()
        self._last_flush  = 0.0
        self.FLUSH_INTERVAL = 10.0

    # ── 외부 호출 메인 함수 ───────────────────────────────
    def send(self, event: dict) -> bool:
        self._try_flush_queue()
        payload = self._build_payload(event)

        # ★ DB 전송 전 터미널 출력
        _print_payload(payload)

        success = self._post(payload)
        if not success:
            print(f"{C.YELLOW}[Sender] 전송 실패 → 큐 저장{C.RESET}")
            self.queue.push(payload)
        return success

    # ── DB 포맷 변환 ──────────────────────────────────────
    # ★ DB 스키마가 바뀌면 이 함수만 수정하면 됩니다 ★
    def _build_payload(self, event: dict) -> dict:
        event_type  = event.get("type", "")
        zone        = event.get("zone", None)
        plate       = event.get("plate", None)
        plate_status = event.get("plate_status", "pending")
        entry_ts    = event.get("entry_time", None)
        park_status = event.get("park_status", "normal")
        linked_zone = event.get("linked_zone", None)
        car_image   = event.get("car_image", None)
        now_ts      = event.get("timestamp", time.time())

        # 구역 미설정 시 null
        # 2칸 주차면 "A-1,A-2" 형태
        if zone is None:
            zone_str = None
        elif linked_zone:
            zone_str = f"{zone},{linked_zone}"
        else:
            zone_str = zone

        def ts_to_str(ts):
            if ts is None or ts == 0:
                return None
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        payload = {
            "event":        _map_event_type(event_type),
            "zone":         zone_str,           # null 허용
            "plate":        plate,              # null 허용
            "plate_status": plate_status,
            "entry_time":   ts_to_str(entry_ts),
            "exit_time":    ts_to_str(now_ts) if event_type == "exit" else None,
            "park_status":  park_status,
            "car_image":    car_image,
        }
        return payload

    # ── 큐 재전송 ─────────────────────────────────────────
    def _try_flush_queue(self):
        now = time.time()
        if self.queue.size() == 0:
            return
        if now - self._last_flush < self.FLUSH_INTERVAL:
            return
        self._last_flush = now
        pending = self.queue.pop_all()
        failed  = []
        for ev in pending:
            if not self._post(ev):
                failed.append(ev)
        for ev in failed:
            self.queue.push(ev)
        if pending:
            sent = len(pending) - len(failed)
            print(f"{C.CYAN}[Sender] Queue flush: "
                  f"{sent}/{len(pending)} sent{C.RESET}")

    # ── HTTP POST ─────────────────────────────────────────
    def _post(self, payload: dict) -> bool:
        try:
            resp = requests.post(
                SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT
            )
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False
        except Exception as e:
            print(f"{C.RED}[Sender] Error: {e}{C.RESET}")
            return False


# ── 터미널 출력 ───────────────────────────────────────────

def _print_payload(payload: dict):
    """
    DB 전송 직전 payload 내용을 터미널에 보기 좋게 출력합니다.
    구역 미설정 시 zone = null 로 표시됩니다.
    """
    event = payload.get("event", "")

    # 이벤트 타입별 색상
    if event == "entry":
        ev_color = C.GREEN
        ev_label = "▶ ENTRY  (입차)"
    elif event == "exit":
        ev_color = C.RED
        ev_label = "◀ EXIT   (출차)"
    elif event == "update":
        ev_color = C.CYAN
        ev_label = "● UPDATE (정보갱신)"
    else:
        ev_color = C.YELLOW
        ev_label = f"? {event}"

    # plate_status 색상
    ps = payload.get("plate_status", "")
    if ps == "confirmed":
        ps_color = C.GREEN
    elif ps == "unreadable":
        ps_color = C.RED
    elif ps == "null":
        ps_color = C.YELLOW
    else:
        ps_color = C.BLUE   # pending

    # park_status 색상
    park = payload.get("park_status", "normal")
    if park == "normal":
        park_color = C.GREEN
    elif park in ["double_park", "aisle_block"]:
        park_color = C.RED
    elif park == "multi_zone":
        park_color = C.YELLOW
    else:
        park_color = C.RESET

    zone      = payload.get("zone")   or "null (구역 미설정)"
    plate     = payload.get("plate")  or "null"
    entry_t   = payload.get("entry_time")  or "null"
    exit_t    = payload.get("exit_time")   or "null"
    car_img   = payload.get("car_image")   or "null"

    print(f"\n{C.BOLD}{'─'*52}{C.RESET}")
    print(f"{C.BOLD}{ev_color}  {ev_label}{C.RESET}")
    print(f"{'─'*52}")
    print(f"  {'이벤트':<12}: {ev_color}{event}{C.RESET}")
    print(f"  {'구역':<12}: {C.BOLD}{zone}{C.RESET}")
    print(f"  {'번호판':<12}: {C.BOLD}{plate}{C.RESET}")
    print(f"  {'번호판상태':<10}: {ps_color}{ps}{C.RESET}")
    print(f"  {'주차형태':<11}: {park_color}{park}{C.RESET}")
    print(f"  {'입차시간':<11}: {entry_t}")
    print(f"  {'출차시간':<11}: {exit_t}")
    print(f"  {'차량이미지':<10}: {car_img}")
    print(f"{'─'*52}\n")


def _map_event_type(raw: str) -> str:
    mapping = {
        "entry":           "entry",
        "exit":            "exit",
        "plate_confirmed": "entry",
        "plate_changed":   "update",
        "plate_update":    "update",
    }
    return mapping.get(raw, raw)