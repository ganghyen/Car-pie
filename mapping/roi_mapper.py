# ============================================================
# [매핑] ROI 매핑 모듈
# main.py 에서 M키로 진입하거나 단독 실행 모두 가능
#
# 단독 실행: cd pie && python -m mapping.roi_mapper
#
# 조작키:
#   Space  : 아루코 마커 인식 -> 가상지도 고정
#   Click  : 가상지도에서 꼭짓점 4점 지정
#   S      : 4점 완성 후 구역 이름 입력 & 저장
#   X      : 구역 이름 입력 후 해당 구역 삭제
#   C      : 점 / 이름 입력 취소
#   E/ESC  : 종료
# ============================================================

import cv2
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.settings import (
        CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
        ARUCO_DICT, ROI_COORDS_PATH,
        VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT,
    )
except ImportError:
    CAMERA_INDEX       = 0
    FRAME_WIDTH        = 1280
    FRAME_HEIGHT       = 720
    ARUCO_DICT         = "DICT_4X4_50"
    ROI_COORDS_PATH    = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "roi_coords.json"
    )
    VIRTUAL_MAP_WIDTH  = 800
    VIRTUAL_MAP_HEIGHT = 600

ARUCO_DICT_MAP = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

WIN_CAM  = "Camera  |  Space: Scan   S: Save   X: Delete   E: Exit"
WIN_VIRT = "Virtual Map  |  Click 4pts  S: Save  X: Delete  C: Cancel  E: Exit"

COL_SAVED   = (80, 200, 80)
COL_CURRENT = (0, 220, 255)
COL_DELETE  = (0, 60, 255)     # 삭제 모드 색상 (빨강)
COL_TEXT    = (255, 255, 255)
COL_WARNING = (0, 60, 255)


class ROIMapper:
    def __init__(self):
        self.H                 = None
        self.H_inv             = None
        self.zones: dict       = {}
        self.current_pts: list = []
        self.zone_counter      = 1
        self.warp_ready        = False
        self.frozen_map: np.ndarray | None = None

        # 입력 모드
        # "none"   → 일반 모드
        # "save"   → S키: 구역 이름 입력 후 저장
        # "delete" → X키: 구역 이름 입력 후 삭제
        self.input_mode: str = "none"
        self.input_text: str = ""

    # ── 저장 / 로드 ───────────────────────────────────────
    def load_existing(self):
        if not os.path.exists(ROI_COORDS_PATH):
            return
        try:
            with open(ROI_COORDS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.zones = data.get("zones", {})
            mat = data.get("homography_matrix")
            if mat:
                self.H        = np.array(mat, dtype=np.float64)
                self.H_inv    = np.linalg.inv(self.H)
                self.warp_ready = True
            if self.zones:
                nums = []
                for k in self.zones:
                    try:
                        nums.append(int(k.split("-")[-1]))
                    except Exception:
                        pass
                if nums:
                    self.zone_counter = max(nums) + 1
            print(f"[ROIMapper] Loaded zones: {list(self.zones.keys())}")
        except Exception as e:
            print(f"[ROIMapper] Load failed (ignored): {e}")
            self.zones = {}

    def _save_to_file(self) -> bool:
        """현재 zones 상태를 파일에 저장합니다."""
        if self.H is None:
            return False
        os.makedirs(os.path.dirname(ROI_COORDS_PATH), exist_ok=True)
        data = {
            "homography_matrix": self.H.tolist(),
            "virtual_map_size":  [VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT],
            "zones":             self.zones,
            "saved_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            with open(ROI_COORDS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[ROIMapper] Save error: {e}")
            return False

    def save_zone(self, zone_name: str) -> bool:
        """현재 4점을 zone_name 으로 저장합니다."""
        if len(self.current_pts) != 4 or self.H is None:
            return False
        self.zones[zone_name] = [p[:] for p in self.current_pts]
        if self._save_to_file():
            print(f"[ROIMapper] Saved: '{zone_name}'")
            return True
        return False

    def delete_zone(self, zone_name: str) -> bool:
        """zone_name 구역을 삭제합니다."""
        if zone_name not in self.zones:
            print(f"[ROIMapper] '{zone_name}' not found")
            return False
        del self.zones[zone_name]
        if self._save_to_file():
            print(f"[ROIMapper] Deleted: '{zone_name}'")
            return True
        return False

    # ── 아루코 인식 + 가상지도 고정 ──────────────────────
    def detect_and_freeze(self, frame: np.ndarray) -> bool:
        adict    = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT_MAP.get(ARUCO_DICT, cv2.aruco.DICT_4X4_50)
        )
        params   = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(adict, params)
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is None or len(ids) < 4:
            n = len(ids) if ids is not None else 0
            print(f"[ROIMapper] Not enough markers: {n}/4")
            return False

        centers = {}
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid in [0, 1, 2, 3]:
                centers[mid] = corners[i][0].mean(axis=0)

        if len(centers) < 4:
            print(f"[ROIMapper] Need ID 0~3, got: {list(centers.keys())}")
            return False

        src = np.float32([centers[0], centers[1], centers[2], centers[3]])
        dst = np.float32([
            [0,                 0                 ],
            [VIRTUAL_MAP_WIDTH, 0                 ],
            [VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT],
            [0,                 VIRTUAL_MAP_HEIGHT],
        ])

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print("[ROIMapper] Homography failed")
            return False

        self.H          = H
        self.H_inv      = np.linalg.inv(H)
        self.warp_ready = True
        self.frozen_map = cv2.warpPerspective(
            frame, self.H, (VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT)
        )
        print("[ROIMapper] Virtual map frozen!")
        return True

    # ── 가상지도 렌더링 ───────────────────────────────────
    def render_virtual(self) -> np.ndarray:
        if self.frozen_map is None:
            canvas = np.zeros((VIRTUAL_MAP_HEIGHT, VIRTUAL_MAP_WIDTH, 3), np.uint8)
            cv2.putText(canvas,
                        "Press SPACE to detect ArUco markers",
                        (30, VIRTUAL_MAP_HEIGHT // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
            cv2.putText(canvas,
                        "Need markers ID 0(TL) 1(TR) 2(BR) 3(BL)",
                        (30, VIRTUAL_MAP_HEIGHT // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 120, 120), 1)
            return canvas

        canvas = self.frozen_map.copy()

        # 저장된 구역 표시
        for zname, pts in self.zones.items():
            poly    = np.array(pts, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly], COL_SAVED)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            cv2.polylines(canvas, [poly], True, COL_SAVED, 2)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            (tw, th), _ = cv2.getTextSize(
                zname, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas,
                          (cx - tw//2 - 4, cy - th - 4),
                          (cx + tw//2 + 4, cy + 6), (0, 0, 0), -1)
            cv2.putText(canvas, zname, (cx - tw//2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_SAVED, 2)

        # 현재 찍는 점/선
        for i, pt in enumerate(self.current_pts):
            cv2.circle(canvas, tuple(pt), 7, COL_CURRENT, -1)
            cv2.putText(canvas, str(i + 1),
                        (pt[0] + 9, pt[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CURRENT, 1)
        if len(self.current_pts) >= 2:
            for i in range(len(self.current_pts) - 1):
                cv2.line(canvas,
                         tuple(self.current_pts[i]),
                         tuple(self.current_pts[i + 1]),
                         COL_CURRENT, 1)
        if len(self.current_pts) == 4:
            cv2.line(canvas,
                     tuple(self.current_pts[3]),
                     tuple(self.current_pts[0]),
                     COL_CURRENT, 1)

        # 상단 안내
        cv2.rectangle(canvas, (0, 0), (VIRTUAL_MAP_WIDTH, 28), (0, 0, 0), -1)

        if self.input_mode == "save":
            # 저장 모드: 초록
            prompt = f"Save as: {self.input_text}_"
            cv2.putText(canvas, prompt, (8, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        elif self.input_mode == "delete":
            # 삭제 모드: 빨강
            prompt = f"Delete zone: {self.input_text}_"
            cv2.putText(canvas, prompt, (8, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_DELETE, 2)

        else:
            n = len(self.current_pts)
            if n < 4:
                guide = f"Click {4-n} more  |  S: save  X: delete  C: cancel  Space: re-scan"
            else:
                guide = "4pts done  |  S: save  |  C: cancel  |  Space: re-scan"
            cv2.putText(canvas, guide, (8, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1)

        # 하단 저장 목록
        saved_str = "Saved: " + (", ".join(self.zones.keys()) if self.zones else "none")
        cv2.putText(canvas, saved_str,
                    (8, VIRTUAL_MAP_HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

        return canvas

    # ── 카메라 창 렌더링 ──────────────────────────────────
    def render_camera(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        adict    = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT_MAP.get(ARUCO_DICT, cv2.aruco.DICT_4X4_50)
        )
        params   = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(adict, params)
        corners, ids, _ = detector.detectMarkers(vis)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cnt   = len(ids)
            color = (0, 255, 0) if cnt >= 4 else (0, 160, 255)
            msg   = "OK - Press SPACE" if cnt >= 4 else "Need 4 markers (ID 0~3)"
            cv2.putText(vis, f"Markers: {cnt}/4  {msg}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            cv2.putText(vis, "No markers detected",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, COL_WARNING, 2)

        status = "READY  |  Space: re-scan" if self.warp_ready else "Space: scan markers"
        color  = (0, 255, 0) if self.warp_ready else (0, 160, 255)
        cv2.putText(vis, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    # ── 마우스 콜백 ───────────────────────────────────────
    def on_mouse(self, event, x, y, flags, param):
        if not self.warp_ready or self.input_mode != "none":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.current_pts) < 4:
                self.current_pts.append([x, y])
                print(f"[ROIMapper] Point {len(self.current_pts)}: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.current_pts = []
            print("[ROIMapper] Points cleared")

    # ── 키 입력 처리 ──────────────────────────────────────
    def handle_key(self, key: int, frame: np.ndarray) -> bool:
        """
        Returns:
            False → 매핑 모드 종료 (E/ESC)
            True  → 계속
        """

        # ── 저장 이름 입력 모드 ───────────────────────────
        if self.input_mode == "save":
            if key == 13:  # Enter
                name = self.input_text.strip()
                if name:
                    overwrite = name in self.zones
                    if self.save_zone(name):
                        action = "overwritten" if overwrite else "added"
                        print(f"[ROIMapper] '{name}' {action}")
                        try:
                            n = int(name.split("-")[-1])
                            self.zone_counter = max(self.zone_counter, n + 1)
                        except Exception:
                            pass
                        self.current_pts = []
                self.input_mode = "none"
                self.input_text = ""

            elif key in [ord('c'), ord('C')]:
                self.input_mode = "none"
                self.input_text = ""
                print("[ROIMapper] Save cancelled")

            elif key == 8:  # Backspace
                self.input_text = self.input_text[:-1]

            elif 32 <= key < 127:
                self.input_text += chr(key)

            return True

        # ── 삭제 이름 입력 모드 ───────────────────────────
        if self.input_mode == "delete":
            if key == 13:  # Enter
                name = self.input_text.strip()
                if name:
                    if self.delete_zone(name):
                        print(f"[ROIMapper] '{name}' deleted")
                    else:
                        print(f"[ROIMapper] '{name}' not found - "
                              f"available: {list(self.zones.keys())}")
                self.input_mode = "none"
                self.input_text = ""

            elif key in [ord('c'), ord('C')]:
                self.input_mode = "none"
                self.input_text = ""
                print("[ROIMapper] Delete cancelled")

            elif key == 8:  # Backspace
                self.input_text = self.input_text[:-1]

            elif 32 <= key < 127:
                self.input_text += chr(key)

            return True

        # ── 일반 모드 ─────────────────────────────────────
        if key == ord(' '):
            self.detect_and_freeze(frame)

        elif key in [ord('s'), ord('S')]:
            if len(self.current_pts) == 4:
                self.input_mode = "save"
                self.input_text = f"A-{self.zone_counter}"
                print(f"[ROIMapper] Enter zone name to save "
                      f"(default: A-{self.zone_counter})")
            else:
                print(f"[ROIMapper] Need 4 points "
                      f"(have {len(self.current_pts)})")

        elif key in [ord('x'), ord('X')]:
            # X키: 삭제할 구역 이름 입력 모드
            if not self.zones:
                print("[ROIMapper] No zones to delete")
            else:
                self.input_mode = "delete"
                self.input_text = ""
                print(f"[ROIMapper] Enter zone name to delete | "
                      f"available: {list(self.zones.keys())}")

        elif key in [ord('c'), ord('C')]:
            self.current_pts = []
            print("[ROIMapper] Points cleared")

        elif key in [ord('e'), ord('E'), 27]:
            return False

        return True

    # ── 단독 실행 루프 ────────────────────────────────────
    def run(self):
        self.load_existing()

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print("[ROIMapper] Camera open failed")
            return

        cv2.namedWindow(WIN_CAM)
        cv2.namedWindow(WIN_VIRT)
        cv2.setMouseCallback(WIN_VIRT, self.on_mouse)

        print("=" * 55)
        print("[ROIMapper] Standalone mode")
        print("  Space : scan markers -> freeze virtual map")
        print("  Click : set 4 corner points on virtual map")
        print("  S     : save zone (enter name)")
        print("  X     : delete zone (enter name)")
        print("  C     : cancel")
        print("  E/ESC : exit")
        print("=" * 55)

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            cv2.imshow(WIN_CAM,  self.render_camera(frame))
            cv2.imshow(WIN_VIRT, self.render_virtual())

            wait_ms = 30 if self.input_mode != "none" else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 255:
                continue

            if not self.handle_key(key, frame):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[ROIMapper] Exit")


if __name__ == "__main__":
    mapper = ROIMapper()
    mapper.run()