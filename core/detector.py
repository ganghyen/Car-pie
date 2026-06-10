# ============================================================
# [Phase 2] YOLOv8 차량 + 번호판 탐지
# ============================================================

import numpy as np
from ultralytics import YOLO
from config.settings import MODEL_PATH, YOLO_CONF, YOLO_IOU, CLASS_CAR, CLASS_PLATE


class VehicleDetector:
    def __init__(self):
        # YOLO 모델 로드
        print(f"[Detector] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print(f"[Detector] Ready | classes: {self.model.names}")

    def detect(self, frame: np.ndarray) -> dict:
        # 프레임에서 차량과 번호판을 탐지하고 결과를 딕셔너리로 반환
        results = self.model(frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)

        cars   = []   # 탐지된 차량 목록
        plates = []   # 탐지된 번호판 목록

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # 클래스 ID와 이름 추출
                cls_id          = int(box.cls[0])
                label           = self.model.names[cls_id]
                # bbox 좌표 (픽셀 단위 정수)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 탐지 신뢰도
                conf            = float(box.conf[0])

                if label == CLASS_CAR:
                    cars.append({
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "conf": conf,
                        # 발바닥 좌표: bbox 하단 중앙 (구역 매핑에 사용)
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
        # 특정 차량 bbox 안에 포함된 번호판 중 신뢰도가 가장 높은 것 반환
        cx1, cy1, cx2, cy2 = car["x1"], car["y1"], car["x2"], car["y2"]
        candidates = []

        for p in plates:
            # 번호판 중심 좌표
            pcx = (p["x1"] + p["x2"]) // 2
            pcy = (p["y1"] + p["y2"]) // 2
            # 번호판 중심이 차량 bbox 안에 있으면 후보로 추가
            # ✅ 번호판이 차량 위쪽으로 나와도 허용 (세로 여유 크게)
            # 가로는 차량 bbox 안에 있어야 함
            y_margin = (cy2 - cy1) * 0.5  # 차량 높이의 50% 여유
            x_margin = 20
            if (cx1 - x_margin <= pcx <= cx2 + x_margin and
                    cy1 - y_margin <= pcy <= cy2):
                candidates.append(p)


        if not candidates:
            return None

        # 후보 중 신뢰도가 가장 높은 번호판 반환
        return max(candidates, key=lambda p: p["conf"])
