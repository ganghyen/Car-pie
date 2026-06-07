# ============================================================
# [Phase 3] 구역 점유 판정 (Overlap Check)
# 차량의 발바닥 좌표가 구역 폴리곤 안에 있는지 검사
# ============================================================

import cv2
import numpy as np


def point_in_zone(point: tuple, zone_polygon: list) -> bool:
    """
    point (vx, vy)가 zone_polygon [[x,y]×4] 안에 있으면 True 반환.
    OpenCV pointPolygonTest: 양수=내부, 0=경계, 음수=외부
    """
    poly = np.array(zone_polygon, dtype=np.float32)
    px, py = float(point[0]), float(point[1])
    # 거리 측정 없이 내/외부 판별만 수행 (measureDist=False)
    result = cv2.pointPolygonTest(poly, (px, py), measureDist=False)
    return result >= 0


def bbox_overlap_ratio(bbox: dict, zone_polygon: list,
                       map_w: int, map_h: int,
                       homography_matrix) -> float:
    """
    이중주차 감지용:
    차량 bbox 4개 꼭짓점을 가상 평면으로 투영한 뒤
    구역 폴리곤과의 교차 면적 / 구역 폴리곤 면적 비율 반환
    """
    if homography_matrix is None:
        return 0.0

    # 카메라 좌표계 bbox 4개 꼭짓점
    cam_corners = np.float32([
        [bbox["x1"], bbox["y1"]],
        [bbox["x2"], bbox["y1"]],
        [bbox["x2"], bbox["y2"]],
        [bbox["x1"], bbox["y2"]],
    ]).reshape(-1, 1, 2)

    # 호모그래피로 가상 평면 좌표로 변환
    virt_corners = cv2.perspectiveTransform(cam_corners, homography_matrix)
    virt_poly = virt_corners.reshape(-1, 2).astype(np.float32)
    zone_poly = np.array(zone_polygon, dtype=np.float32)

    try:
        # Sutherland-Hodgman으로 교차 면적 계산
        inter_area = _polygon_intersection_area(virt_poly, zone_poly)
        zone_area = cv2.contourArea(zone_poly)
        if zone_area == 0:
            return 0.0
        # 교차 면적 / 구역 면적 = 겹침 비율
        return inter_area / zone_area
    except Exception:
        return 0.0


def _polygon_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Sutherland-Hodgman 알고리즘으로 두 볼록 폴리곤의 교차 면적 계산."""

    def clip(subject, clip_polygon):
        # 클리핑 폴리곤의 각 엣지에 대해 subject 폴리곤을 클리핑
        output = list(subject)
        if not output:
            return output
        for i in range(len(clip_polygon)):
            if not output:
                return output
            input_list = output
            output = []
            edge_start = clip_polygon[i]
            edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
            for j in range(len(input_list)):
                current = input_list[j]
                previous = input_list[j - 1]
                if _inside(current, edge_start, edge_end):
                    # current만 안쪽: previous가 바깥이면 교점 추가
                    if not _inside(previous, edge_start, edge_end):
                        output.append(_intersection(previous, current, edge_start, edge_end))
                    output.append(current)
                elif _inside(previous, edge_start, edge_end):
                    # previous만 안쪽: 교점만 추가
                    output.append(_intersection(previous, current, edge_start, edge_end))
        return output

    def _inside(p, a, b):
        # 점 p가 엣지 a→b의 왼쪽(안쪽)에 있는지 판별
        return (b[0] - a[0]) * (p[1] - a[1]) > (b[1] - a[1]) * (p[0] - a[0])

    def _intersection(p1, p2, p3, p4):
        # 선분 p1-p2와 선분 p3-p4의 교점 계산
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return p1  # 평행한 경우 첫 번째 점 반환
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    # Sutherland-Hodgman 알고리즘 실행
    clipped = clip(
        [tuple(p) for p in poly1],
        [tuple(p) for p in poly2]
    )

    if len(clipped) < 3:
        return 0.0

    # 신발끈 공식(Shoelace formula)으로 폴리곤 면적 계산
    n = len(clipped)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += clipped[i][0] * clipped[j][1]
        area -= clipped[j][0] * clipped[i][1]
    return abs(area) / 2.0