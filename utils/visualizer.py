# ============================================================
# [유틸] 화면 출력
# ============================================================

import cv2
import numpy as np
import time
from state.zone_state import ZoneStatus, ParkStatus, PlateStatus
from config.settings import (
    COLOR_EMPTY, COLOR_OCCUPIED, COLOR_TIMEOUT,
    COLOR_BBOX_CAR, COLOR_BBOX_PLATE, COLOR_AISLE_WARN,
    AISLE_ZONE_PREFIX,
    STILL_SECONDS_REQUIRED, EXIT_TIMEOUT_SECONDS,
    AISLE_STILL_SECONDS,
)


class Visualizer:
    def draw_frame(self, frame: np.ndarray,
                   cars: list[dict],
                   plates: list[dict],
                   zone_statuses: dict,
                   homography_transformer,
                   fps: float = 0.0,
                   state_machine=None) -> np.ndarray:

        vis = frame.copy()

        # ── 차량 bbox ─────────────────────────────────────
        for car in cars:
            x1, y1, x2, y2 = car["x1"], car["y1"], car["x2"], car["y2"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BBOX_CAR, 2)
            cv2.putText(vis, f"car {car['conf']:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BBOX_CAR, 1)
            cv2.circle(vis, (car["foot_x"], car["foot_y"]),
                       4, (0, 255, 255), -1)

        # ── 번호판 bbox ───────────────────────────────────
        for plate in plates:
            x1, y1, x2, y2 = plate["x1"], plate["y1"], plate["x2"], plate["y2"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BBOX_PLATE, 1)
            cv2.putText(vis, f"plate {plate['conf']:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BBOX_PLATE, 1)

        # ── 구역 오버레이 ─────────────────────────────────
        if homography_transformer.is_ready():
            for zone_name, info in zone_statuses.items():
                zone_pts = homography_transformer.zones.get(zone_name, [])
                if not zone_pts:
                    continue

                cam_pts = [
                    homography_transformer.virtual_to_camera(tuple(p))
                    for p in zone_pts
                ]
                cam_pts_int = np.array(
                    [(int(p[0]), int(p[1])) for p in cam_pts],
                    dtype=np.int32
                )

                status      = info["status"]
                park_status = info.get("park_status", "normal")
                is_aisle    = zone_name.startswith(AISLE_ZONE_PREFIX)

                if is_aisle and status == ZoneStatus.OCCUPIED.value:
                    color = COLOR_AISLE_WARN
                elif status == ZoneStatus.OCCUPIED.value:
                    color = COLOR_OCCUPIED
                elif status == ZoneStatus.TIMEOUT.value:
                    color = COLOR_TIMEOUT
                else:
                    color = COLOR_EMPTY

                overlay = vis.copy()
                cv2.fillPoly(overlay, [cam_pts_int], color)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.polylines(vis, [cam_pts_int], True, color, 2)

                cx = int(np.mean([p[0] for p in cam_pts]))
                cy = int(np.mean([p[1] for p in cam_pts]))

                # 구역 이름
                cv2.putText(vis, zone_name,
                            (cx - 20, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                # ★ 번호판 텍스트 화면 표시 제거
                # (터미널에서만 확인)

                # 타이머
                if state_machine:
                    _draw_zone_timer(
                        vis, zone_name, info, state_machine,
                        cx, cy, color, is_aisle
                    )

                # 주차 형태 태그
                tag = ""
                if park_status == ParkStatus.DOUBLE_PARK.value:
                    tag = "DOUBLE PARK"
                elif park_status == ParkStatus.MULTI_ZONE.value:
                    tag = f"MULTI({info.get('linked_zone','')})"
                elif park_status == ParkStatus.AISLE_BLOCK.value:
                    tag = "AISLE BLOCK!"

                if tag:
                    cv2.putText(vis, tag,
                                (cx - 40, cy + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)

        # ── FPS ───────────────────────────────────────────
        cv2.putText(vis, f"FPS: {fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # ── 우측 상단 상태 요약 (구역+상태만) ─────────────
        y_off = 25
        for zone_name, info in zone_statuses.items():
            status = info["status"]
            color  = (COLOR_OCCUPIED if status == "occupied"
                      else COLOR_TIMEOUT if status == "timeout"
                      else COLOR_EMPTY)
            # ★ 번호판 제거하고 구역 + 상태만 표시
            txt = f"{zone_name}: {status.upper()}"
            cv2.putText(vis, txt,
                        (frame.shape[1] - 200, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_off += 18

        return vis


def _draw_zone_timer(vis, zone_name, info, state_machine,
                     cx, cy, color, is_aisle):
    zone = state_machine.zones.get(zone_name)
    if zone is None:
        return

    now           = time.time()
    still_required = AISLE_STILL_SECONDS if is_aisle else STILL_SECONDS_REQUIRED
    status        = info["status"]

    # EMPTY: 정지 타이머
    if status == ZoneStatus.EMPTY.value:
        if zone.is_still and zone.still_since > 0:
            elapsed = min(now - zone.still_since, still_required)
            ratio   = elapsed / still_required

            cv2.putText(vis, f"{elapsed:.1f}s / {still_required:.0f}s",
                        (cx - 35, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 180), 2)

            bar_w = 80
            bar_h = 8
            bar_x = cx - bar_w // 2
            bar_y = cy + 22

            cv2.rectangle(vis,
                          (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (80, 80, 80), -1)

            filled_w  = int(bar_w * ratio)
            bar_color = ((0, 220, 0) if ratio < 0.5
                         else (0, 200, 255) if ratio < 0.8
                         else (0, 100, 255))

            if filled_w > 0:
                cv2.rectangle(vis,
                              (bar_x, bar_y),
                              (bar_x + filled_w, bar_y + bar_h),
                              bar_color, -1)

            cv2.rectangle(vis,
                          (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (200, 200, 200), 1)

    # OCCUPIED: PARKED
    elif status == ZoneStatus.OCCUPIED.value:
        cv2.putText(vis, "PARKED",
                    (cx - 28, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (100, 255, 100), 2)

    # TIMEOUT: 카운트다운
    elif status == ZoneStatus.TIMEOUT.value:
        if zone.timeout_start > 0:
            remaining = max(0, EXIT_TIMEOUT_SECONDS - (now - zone.timeout_start))
            cv2.putText(vis, f"EXIT? {remaining:.1f}s",
                        (cx - 35, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR_TIMEOUT, 2)