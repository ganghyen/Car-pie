# ============================================================
# [설정] 전체 시스템 상수 / 파라미터
# ============================================================

import os

# 현재 파일 기준으로 프로젝트 루트 경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 경로 ──────────────────────────────────────────────────
# YOLO 모델 파일 경로
MODEL_PATH        = os.path.join(BASE_DIR, "data", "models", "mini_yolo.pt")
# ROI 구역 좌표 및 호모그래피 행렬 저장 파일
ROI_COORDS_PATH   = os.path.join(BASE_DIR, "data", "roi_coords.json")
# 입출차 스냅샷 이미지 저장 폴더
SNAPSHOT_DIR      = os.path.join(BASE_DIR, "data", "snapshots")
# 전원 차단 복구용 상태 백업 파일
STATE_BACKUP_PATH = os.path.join(BASE_DIR, "data", "state_backup.json")
# 상태 백업 주기 (초)
STATE_BACKUP_INTERVAL = 60.0

# ── 카메라 ─────────────────────────────────────────────────
# 카메라 장치 번호 (0 = 첫 번째 카메라)
CAMERA_INDEX      = 0
# 캡처 해상도 너비
FRAME_WIDTH       = 1280
# 캡처 해상도 높이
FRAME_HEIGHT      = 720
# 목표 프레임레이트
TARGET_FPS        = 15

# ── YOLO 탐지 ──────────────────────────────────────────────
# YOLO 탐지 신뢰도 임계값 (이 값 이상만 탐지로 인정)
YOLO_CONF         = 0.2
# YOLO IoU 임계값 (겹치는 박스 제거 기준)
YOLO_IOU          = 0.4
# YOLO 클래스 이름: 차량
CLASS_CAR         = "car"
# YOLO 클래스 이름: 번호판
CLASS_PLATE       = "plate"

# ── 구역 상태 머신 ─────────────────────────────────────────
# 차량 정지 판정 픽셀 임계값 (발바닥 좌표 이동량이 이 값 미만이면 정지로 판단)
STILL_PIXEL_THRESHOLD  = 8
# 차량이 정지 상태를 유지해야 입차로 확정하는 시간 (초)
STILL_SECONDS_REQUIRED = 5.0
# 차가 안 보인 후 출차로 확정하기까지 대기 시간 (초)
EXIT_TIMEOUT_SECONDS   = 5
# 번호판 재인식 주기 (초)
RECHECK_INTERVAL_SEC   = 30

# 통로 구역 정지 판정 시간 (초)
AISLE_STILL_SECONDS    = 3.0
# 통로 구역 이름 접두사 (이 문자로 시작하면 통로 구역으로 판단)
AISLE_ZONE_PREFIX      = "P"

# ── 픽셀 비교 ──────────────────────────────────────────────
# 빈 구역 스냅샷과 현재 이미지 비교 시 픽셀 변화로 인정하는 최소 차이값
PIXEL_DIFF_THRESHOLD            = 30
# 변화된 픽셀 비율이 이 값을 넘으면 차가 있다고 판단
PIXEL_CHECK_OCCUPIED            = 0.35
# 조명 변화로 판단하는 밝기 차이 임계값
PIXEL_LIGHTING_CHANGE_THRESHOLD = 20
# 조명 변화 감지 체크 주기 (초)
PIXEL_LIGHTING_UPDATE_INTERVAL  = 30.0

# ── 2칸 주차 판정 ──────────────────────────────────────────
# 차량 bbox가 구역과 겹치는 비율이 이 값 이상이면 해당 구역 점유로 판정
# 0.30 → 0.15 로 완화 (더 민감하게 탐지)
MULTI_ZONE_OVERLAP_RATIO = 0.15

# ── OCR ────────────────────────────────────────────────────
# 번호판 인식 샘플 횟수 (투표용)
OCR_SAMPLE_COUNT         = 3
# 샘플 간 인식 간격 (초)
OCR_SAMPLE_INTERVAL      = 0.4
# OCR 결과 신뢰도 최소 임계값
OCR_CONF_THRESHOLD       = 0.35
# 인식된 텍스트 최소 길이 (이 미만이면 무효 처리)
OCR_MIN_TEXT_LENGTH      = 4
# 번호판 crop 시 상하좌우 여백 (픽셀)
PLATE_PADDING            = 10
# 번호판 이미지 업스케일 배율
PLATE_UPSCALE            = 3.0
# 동시 실행 OCR 스레드 최대 수
OCR_MAX_THREADS          = 2
# 연속 OCR 실패 횟수 한도 (이 횟수 초과 시 UNREADABLE 처리)
OCR_FAIL_LIMIT           = 2
# UNREADABLE 판정 후 재시도까지 대기 시간 (초)
OCR_UNREADABLE_RETRY_SEC = 60.0

# ── 이물질 / 흐림 감지 ────────────────────────────────────
# 라플라시안 분산 기반 선명도 임계값 (이 미만이면 흐림으로 판단)
BLUR_DETECT_THRESHOLD    = 80.0
# 흐림 체크 주기 (초)
BLUR_CHECK_INTERVAL      = 3.0
# 연속 흐림 감지 횟수가 이 값 이상이면 카메라 오염으로 확정
BLUR_CONFIRM_COUNT       = 3

# ── 통신 ──────────────────────────────────────────────────
# FastAPI 서버 이벤트 수신 엔드포인트 주소
SERVER_URL      = "http://172.20.10.5:8000/api/event"
# 현재 카메라 장비가 속한 아파트 번호
# 여러 아파트를 운영하면 장비별로 이 값을 바꿔서 사용
APARTMENT_NO    = 1
# HTTP 요청 타임아웃 (초)
REQUEST_TIMEOUT = 15
# 전송 실패 시 로컬에 임시 저장하는 큐 파일 경로
QUEUE_FILE_PATH = os.path.join(BASE_DIR, "data", "pending_queue.json")

# ── 스냅샷 자동 삭제 ───────────────────────────────────────
# 스냅샷 보관 최대 시간 (시간 단위)
SNAPSHOT_MAX_AGE_HOURS    = 12
# 스냅샷 정리 주기 (초)
SNAPSHOT_CLEANUP_INTERVAL = 1800.0

# ── 카메라 흔들림 ─────────────────────────────────────────
# 아루코 마커 이동량이 이 값 이상이면 흔들림으로 판단 (픽셀)
CAMERA_SHAKE_THRESHOLD      = 15
# 흔들림 감지 체크 주기 (초)
CAMERA_SHAKE_CHECK_INTERVAL = 5.0

# ── 매핑 ──────────────────────────────────────────────────
# 아루코 마커 딕셔너리 종류
ARUCO_DICT         = "DICT_4X4_50"
# 실제 주차장 가로 크기 (cm)
REAL_WIDTH_CM      = 50
# 실제 주차장 세로 크기 (cm)
REAL_HEIGHT_CM     = 40
# 가상 지도 1픽셀당 실제 크기 (cm)
CM_PER_PIXEL       = 16
# 가상 지도 너비 (픽셀)
VIRTUAL_MAP_WIDTH  = REAL_WIDTH_CM  * CM_PER_PIXEL
# 가상 지도 높이 (픽셀)
VIRTUAL_MAP_HEIGHT = REAL_HEIGHT_CM * CM_PER_PIXEL

# ── 시각화 색상 (BGR) ──────────────────────────────────────
# 빈 구역 표시 색 (초록)
COLOR_EMPTY       = (100, 220, 100)
# 점유 구역 표시 색 (파랑)
COLOR_OCCUPIED    = (60,  60,  220)
# 타임아웃 구역 표시 색 (주황)
COLOR_TIMEOUT     = (30,  165, 255)
# 차량 bbox 색 (노랑)
COLOR_BBOX_CAR    = (255, 200,   0)
# 번호판 bbox 색 (하늘)
COLOR_BBOX_PLATE  = (0,   200, 255)
# 통로 경고 색 (빨강)
COLOR_AISLE_WARN  = (0,    60, 255)
