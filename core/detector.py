# ============================================================
# [Phase 2] YOLOv8 차량 + 번호판 탐지
# ============================================================

import numpy as np
from ultralytics import YOLO
from config.settings import MODEL_PATH, YOLO_CONF, YOLO_IOU, CLASS_CAR, CLASS_PLATE


class VehicleDetector:
    def __init__(self):
        print(f"[Detector] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print(f"[Detector] Ready | classes: {self.model.names}")

    def detect(self, frame: np.ndarray) -> dict:
        results = self.model(frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)

        cars   = []
        plates = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id          = int(box.cls[0])
                label           = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf            = float(box.conf[0])

                if label == CLASS_CAR:
                    cars.append({
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "conf": conf,
                        "foot_x": (x1 + x2) // 2,
                        "foot_y": y2,
                    })
                elif label == CLASS_PLATE:
                    plates.append({
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "conf": conf,
                    })

        return {"cars": cars, "plates": plates}

    def find_plate_for_car(self, car: dict, plates: list) -> dict | None:
        cx1, cy1, cx2, cy2 = car["x1"], car["y1"], car["x2"], car["y2"]
        candidates = []
        for p in plates:
            pcx = (p["x1"] + p["x2"]) // 2
            pcy = (p["y1"] + p["y2"]) // 2
            if cx1 <= pcx <= cx2 and cy1 <= pcy <= cy2:
                candidates.append(p)
        if not candidates:
            return None
        return max(candidates, key=lambda p: p["conf"])