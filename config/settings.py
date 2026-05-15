# ============================================================
# [설정] 전체 시스템 상수 / 파라미터
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 경로 ──────────────────────────────────────────────────
MODEL_PATH        = os.path.join(BASE_DIR, "data", "models", "minicar_yolo.pt")
ROI_COORDS_PATH   = os.path.join(BASE_DIR, "data", "roi_coords.json")
SNAPSHOT_DIR      = os.path.join(BASE_DIR, "data", "snapshots")
STATE_BACKUP_PATH = os.path.join(BASE_DIR, "data", "state_backup.json")
STATE_BACKUP_INTERVAL = 60.0

# ── 카메라 ─────────────────────────────────────────────────
CAMERA_INDEX      = 0
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720
TARGET_FPS        = 15

# ── YOLO 탐지 ──────────────────────────────────────────────
YOLO_CONF         = 0.45
YOLO_IOU          = 0.4
CLASS_CAR         = "car"
CLASS_PLATE       = "plate"

# ── 구역 상태 머신 ─────────────────────────────────────────
STILL_PIXEL_THRESHOLD  = 8
STILL_SECONDS_REQUIRED = 3.0
EXIT_TIMEOUT_SECONDS   = 8
RECHECK_INTERVAL_SEC   = 90

AISLE_STILL_SECONDS    = 3.0
AISLE_ZONE_PREFIX      = "P"

# ── 픽셀 비교 ──────────────────────────────────────────────
PIXEL_DIFF_THRESHOLD            = 30
PIXEL_CHECK_OCCUPIED            = 0.15
PIXEL_LIGHTING_CHANGE_THRESHOLD = 20
PIXEL_LIGHTING_UPDATE_INTERVAL  = 30.0

# ── 2칸 주차 판정 ──────────────────────────────────────────
MULTI_ZONE_OVERLAP_RATIO = 0.30

# ── OCR ────────────────────────────────────────────────────
OCR_SAMPLE_COUNT         = 5
OCR_SAMPLE_INTERVAL      = 0.4
OCR_CONF_THRESHOLD       = 0.35  # 낮은 confidence 투표 참여 방지
OCR_MIN_TEXT_LENGTH      = 4
PLATE_PADDING            = 10
PLATE_UPSCALE            = 3.0
OCR_MAX_THREADS          = 2
OCR_FAIL_LIMIT           = 3
OCR_UNREADABLE_RETRY_SEC = 60.0

# ── 이물질 / 흐림 감지 ────────────────────────────────────
# 카메라 렌즈 이물질 감지
# Laplacian variance 로 프레임 선명도 측정
# 값이 낮을수록 흐림 (100 이하 = 흐림)
# 실내 조명 환경에서 정상 값은 보통 200~500 수준
# 테스트 후 환경에 맞게 조정하세요
BLUR_DETECT_THRESHOLD    = 80.0   # 이 값 이하면 흐림으로 판단
BLUR_CHECK_INTERVAL      = 3.0    # 초마다 선명도 체크
BLUR_CONFIRM_COUNT       = 3      # 연속 N회 흐림 감지 시 경고
                                  # (일시적 흔들림 오판 방지)

# 번호판 이물질 (기존 OCR 실패 로직과 연동)
# OCR_FAIL_LIMIT 회 연속 실패 시 PLATE_UNREADABLE 처리
# → 이미 reader.py 에 구현되어 있음

# ── 통신 ──────────────────────────────────────────────────
SERVER_URL        = "http://localhost:3000/api/parking"
REQUEST_TIMEOUT   = 3
QUEUE_FILE_PATH   = os.path.join(BASE_DIR, "data", "pending_queue.json")

# ── 스냅샷 자동 삭제 ───────────────────────────────────────
SNAPSHOT_MAX_AGE_HOURS    = 12
SNAPSHOT_CLEANUP_INTERVAL = 1800.0

# ── 카메라 흔들림 ─────────────────────────────────────────
CAMERA_SHAKE_THRESHOLD      = 15
CAMERA_SHAKE_CHECK_INTERVAL = 5.0

# ── 매핑 ──────────────────────────────────────────────────
ARUCO_DICT         = "DICT_4X4_50"
# ── 실제 주차장 크기 (cm) ────────────────────────────────
# 아루코 마커 안쪽 꼭짓점 기준 실제 거리
# 마커0-마커1 가로: 50cm
# 마커0-마커3 세로: 40cm
REAL_WIDTH_CM      = 50     # 가로 실제 거리 (cm)
REAL_HEIGHT_CM     = 40     # 세로 실제 거리 (cm)

# 1cm 당 픽셀 수 (높을수록 가상지도 정밀도 높아짐)
# 라즈베리파이 2GB 기준 16 권장 (너무 높으면 부하)
CM_PER_PIXEL       = 16

# 가상지도 크기 자동 계산
VIRTUAL_MAP_WIDTH  = REAL_WIDTH_CM  * CM_PER_PIXEL   # 50 × 16 = 800
VIRTUAL_MAP_HEIGHT = REAL_HEIGHT_CM * CM_PER_PIXEL   # 40 × 16 = 640

# ── 시각화 ─────────────────────────────────────────────────
COLOR_EMPTY       = (100, 220, 100)
COLOR_OCCUPIED    = (60,  60,  220)
COLOR_TIMEOUT     = (30,  165, 255)
COLOR_BBOX_CAR    = (255, 200,   0)
COLOR_BBOX_PLATE  = (0,   200, 255)
COLOR_AISLE_WARN  = (0,    60, 255)