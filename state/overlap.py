# ============================================================
# [Phase 3] 구역 점유 판정 (Overlap Check)
# 차량의 발바닥 좌표가 가상 평면의 구역 폴리곤 안에 있는지 검사합니다.
# ============================================================

import cv2
import numpy as np


def point_in_zone(point: tuple, zone_polygon: list) -> bool:
    """
    point (vx, vy) 가 zone_polygon [[x,y]×4] 안에 있으면 True.
    OpenCV pointPolygonTest 사용: 양수=내부, 0=경계, 음수=외부
    """
    poly = np.array(zone_polygon, dtype=np.float32)
    px, py = float(point[0]), float(point[1])
    result = cv2.pointPolygonTest(poly, (px, py), measureDist=False)
    return result >= 0


def bbox_overlap_ratio(bbox: dict, zone_polygon: list,
                       map_w: int, map_h: int,
                       homography_matrix) -> float:
    """
    이중주차 감지용: 차량 bbox의 가상 평면 투영 면적과
    구역 폴리곤의 겹침 비율을 반환합니다.

    bbox의 4개 꼭짓점을 가상 평면으로 변환한 뒤
    구역 폴리곤과의 교차 면적 / 구역 폴리곤 면적 을 계산합니다.
    """
    if homography_matrix is None:
        return 0.0

    # bbox 4 꼭짓점 → 가상 평면 투영
    cam_corners = np.float32([
        [bbox["x1"], bbox["y1"]],
        [bbox["x2"], bbox["y1"]],
        [bbox["x2"], bbox["y2"]],
        [bbox["x1"], bbox["y2"]],
    ]).reshape(-1, 1, 2)

    virt_corners = cv2.perspectiveTransform(cam_corners, homography_matrix)
    virt_poly = virt_corners.reshape(-1, 2).astype(np.float32)

    zone_poly = np.array(zone_polygon, dtype=np.float32)

    # Shapely 없이 OpenCV로 교차 면적 계산
    try:
        inter_area = _polygon_intersection_area(virt_poly, zone_poly)
        zone_area = cv2.contourArea(zone_poly)
        if zone_area == 0:
            return 0.0
        return inter_area / zone_area
    except Exception:
        return 0.0


def _polygon_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Sutherland-Hodgman 알고리즘으로 두 볼록 폴리곤의 교차 면적 계산."""
    def clip(subject, clip_polygon):
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
                    if not _inside(previous, edge_start, edge_end):
                        output.append(_intersection(previous, current, edge_start, edge_end))
                    output.append(current)
                elif _inside(previous, edge_start, edge_end):
                    output.append(_intersection(previous, current, edge_start, edge_end))
        return output

    def _inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) > (b[1] - a[1]) * (p[0] - a[0])

    def _intersection(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    clipped = clip(
        [tuple(p) for p in poly1],
        [tuple(p) for p in poly2]
    )
    if len(clipped) < 3:
        return 0.0

    # 신발끈 공식으로 면적 계산
    n = len(clipped)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += clipped[i][0] * clipped[j][1]
        area -= clipped[j][0] * clipped[i][1]
    return abs(area) / 2.0