# ============================================================
# [메인] 스마트 주차 관리 시스템
# 수정사항:
#   1. send_queue → PriorityQueue (출차>입차>업데이트 순서 보장)
#   2. 번호판 안 보여도 null 즉시 전송 안함 → OCR 시도 후 전송
# ============================================================

import cv2
import time
import sys
import os
import json
import threading
import queue
import numpy as np
from datetime import datetime, timedelta

# 현재 파일 디렉토리를 모듈 검색 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    ROI_COORDS_PATH, SNAPSHOT_DIR,
    VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT,
    CAMERA_SHAKE_CHECK_INTERVAL,
    SNAPSHOT_MAX_AGE_HOURS, SNAPSHOT_CLEANUP_INTERVAL,
    OCR_MAX_THREADS,
    MULTI_ZONE_OVERLAP_RATIO,
    STATE_BACKUP_PATH, STATE_BACKUP_INTERVAL,
)
from mapping.homography import HomographyTransformer
from mapping.roi_mapper import ROIMapper
from core.preprocessor import Preprocessor
from core.detector import VehicleDetector
from state.overlap import point_in_zone
from state.zone_state import (
    ParkingStateMachine, ZoneStatus, PlateStatus
)
from ocr.reader import PlateReader, PLATE_UNREADABLE
from comm.sender import EventSender
from utils.logger import get_logger
from utils.visualizer import Visualizer

logger = get_logger("parking")

# 창 이름 상수
WIN_MAIN = "Smart Parking  |  M: Mapping   Q: Quit"
WIN_VIRT = "Virtual Map  |  Click 4pts  S: Save  X: Delete  C: Cancel  E: Exit"

# ── PriorityQueue 우선순위 상수 ───────────────────────────
PRIORITY_EXIT   = 1  # 출차 최우선 (DB 불일치 방지)
PRIORITY_ENTRY  = 2  # 입차
PRIORITY_UPDATE = 3  # 번호판 업데이트 (가장 낮은 우선순위)


class OcrTask:
    """OCR 작업 단위: 구역명, 스냅샷, bbox 정보, 원본 입차 이벤트 묶음."""
    def __init__(self, zone_name, snapshot, car_bbox,
                 plate_bbox, entry_event):
        self.zone_name   = zone_name    # 처리할 구역 이름
        self.snapshot    = snapshot     # 입차 시점 캡처 프레임
        self.car_bbox    = car_bbox     # 차량 bbox 좌표
        self.plate_bbox  = plate_bbox   # 번호판 bbox 좌표 (없으면 None)
        self.entry_event = entry_event  # OCR 완료 후 전송할 입차 이벤트
        self.queued_at   = time.time()  # 큐 등록 시각 (디버깅용)


def ocr_worker(ocr_queue: queue.Queue,
               send_queue: queue.PriorityQueue,
               ocr_reader: PlateReader,
               state_machine: ParkingStateMachine,
               stop_event: threading.Event):
    """
    OCR 전담 스레드.
    ocr_queue에서 작업을 꺼내 번호판 인식 후
    결과를 entry 이벤트에 붙여서 send_queue에 넣음.
    """
    logger.info("[OCR Worker] 시작")

    while not stop_event.is_set():
        try:
            # 1초 대기 후 작업 없으면 루프 재시작 (stop_event 체크)
            task: OcrTask = ocr_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            zone_name = task.zone_name

            # 스냅샷에서 번호판 인식 (다수결 투표)
            plate = ocr_reader.vote_from_snapshot(
                snapshot_frame=task.snapshot,
                bbox=task.car_bbox,
                zone_name=zone_name,
                plate_bbox=task.plate_bbox,
            )

            # 인식 결과를 구역 상태에 반영
            state_machine.set_plate(zone_name, plate)
            zone = state_machine.zones.get(zone_name)
            ps   = zone.plate_status.value if zone else "null"

            logger.info(f"[OCR Worker] {zone_name} 완료: {plate} ({ps})")

            # 입차 이벤트에 번호판 정보 추가
            entry_event                 = task.entry_event
            entry_event["plate"]        = zone.plate if zone else plate
            entry_event["plate_status"] = ps

            # 입차 이벤트를 우선순위 2로 send_queue에 등록
            send_queue.put_nowait((PRIORITY_ENTRY, entry_event))

            if zone and zone.plate_status == PlateStatus.UNREADABLE:
                logger.warning(f"[UNREADABLE] {zone_name} 번호판 인식 불가")

        except Exception as e:
            logger.error(f"[OCR Worker] {task.zone_name} 오류: {e}")
        finally:
            # 작업 완료 표시 (queue.join() 사용 시 필요)
            ocr_queue.task_done()

    logger.info("[OCR Worker] 종료")


def send_worker(send_queue: queue.PriorityQueue,
                sender: EventSender,
                stop_event: threading.Event):
    """
    전송 전담 스레드.
    PriorityQueue에서 우선순위 순으로 꺼내 FastAPI 서버로 전송.
    출차(1) > 입차(2) > 업데이트(3) 순서 보장.
    """
    logger.info("[Send Worker] 시작")

    while not stop_event.is_set():
        try:
            # (우선순위, 이벤트) 튜플로 꺼냄
            priority, event = send_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            sender.send(event)
        except Exception as e:
            logger.error(f"[Send Worker] 전송 오류: {e}")
        finally:
            send_queue.task_done()

    logger.info("[Send Worker] 종료")


def main():
    logger.info("=" * 50)
    logger.info("Smart Parking System Start")
    logger.info("=" * 50)

    # 필요한 디렉토리 생성
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_BACKUP_PATH), exist_ok=True)

    # ── 모듈 초기화 ────────────────────────────────────────
    homography   = HomographyTransformer()  # 호모그래피 좌표 변환
    preprocessor = Preprocessor()           # CLAHE 전처리 + 흐림 감지
    detector     = VehicleDetector()        # YOLO 탐지
    ocr_reader   = PlateReader()            # EasyOCR 번호판 인식
    sender       = EventSender()            # FastAPI 전송
    visualizer   = Visualizer()             # 화면 시각화
    mapper       = ROIMapper()              # 구역 매핑 도구

    # 저장된 호모그래피 및 구역 정보 로드
    homography.load()
    mapper.load_existing()

    if not homography.is_ready():
        logger.warning("No mapping - Press M to enter mapping mode")
    else:
        logger.info(f"Mapping loaded | zones: {list(homography.zones.keys())}")

    # 구역 목록으로 상태 머신 초기화
    zone_keys     = list(homography.zones.keys()) if homography.zones else []
    state_machine = ParkingStateMachine(zone_keys)

    # 전원 차단 복구: 60분 이내 백업 상태 복원
    _restore_state(state_machine)

    # 매핑 파일 변경 감지용 변수
    last_mtime     = _get_mtime(ROI_COORDS_PATH)
    CHECK_INTERVAL = 2.0   # 파일 변경 체크 주기 (초)
    last_check     = time.time()

    # ── 카메라 초기화 ──────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    # C920 웹캠 최적화 설정
    cap.set(cv2.CAP_PROP_AUTOFOCUS,     1)      # 자동 포커스 활성화
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 자동 노출 모드
    cap.set(cv2.CAP_PROP_EXPOSURE,      -3)     # 노출값
    cap.set(cv2.CAP_PROP_BRIGHTNESS,    100)    # 밝기
    cap.set(cv2.CAP_PROP_CONTRAST,      150)    # 대비
    cap.set(cv2.CAP_PROP_SHARPNESS,     200)    # 선명도

    # 카메라 자동 포커스 안정화 대기
    time.sleep(5.0)
    logger.info("[Camera] C920 자동 포커스 설정 완료")

    if not cap.isOpened():
        logger.error(f"Camera {CAMERA_INDEX} open failed")
        sys.exit(1)

    # ── 작업 큐 초기화 ─────────────────────────────────────
    ocr_queue  = queue.Queue(maxsize=20)           # OCR 작업 큐
    send_queue = queue.PriorityQueue(maxsize=50)   # 우선순위 전송 큐
    stop_event = threading.Event()                  # 스레드 종료 신호

    # ── OCR 워커 스레드 시작 ───────────────────────────────
    ocr_workers = []
    for i in range(OCR_MAX_THREADS):
        t = threading.Thread(
            target=ocr_worker,
            args=(ocr_queue, send_queue, ocr_reader,
                  state_machine, stop_event),
            daemon=True,
            name=f"OCR-Worker-{i+1}"
        )
        t.start()
        ocr_workers.append(t)

    # ── 전송 워커 스레드 시작 ──────────────────────────────
    send_thread = threading.Thread(
        target=send_worker,
        args=(send_queue, sender, stop_event),
        daemon=True,
        name="Send-Worker"
    )
    send_thread.start()

    logger.info(f"[Workers] OCR x{OCR_MAX_THREADS}, Send x1 시작")

    # OCR 중복 제출 방지: 구역별 OCR 제출 여부
    ocr_submitted: dict[str, bool] = {}
    # OCR 완료 전 출차 발생 시 처리용: 구역별 미전송 입차 이벤트
    pending_entry: dict[str, dict] = {}

    # 주기별 작업 타이머
    last_shake_check  = time.time()   # 카메라 흔들림 체크
    last_snap_cleanup = time.time()   # 스냅샷 파일 정리
    last_state_backup = time.time()   # 상태 백업

    # 흔들림 상태 메시지 (화면 표시용)
    shake_status_msg   = ""
    shake_status_time  = 0.0
    STATUS_DISPLAY_SEC = 3.0  # 상태 메시지 표시 시간

    # 빈 구역 스냅샷 초기화 완료 여부
    empty_snap_initialized = False
    mapping_mode  = False   # 매핑 모드 활성화 여부
    virt_win_open = False   # 가상 지도 창 열림 여부

    cv2.namedWindow(WIN_MAIN)
    prev_time = time.time()
    logger.info("Camera started | M: mapping  Q/ESC: quit")

    # ── 메인 루프 ──────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            # 프레임 읽기 실패 시 잠시 대기 후 재시도
            time.sleep(0.03)
            continue

        now = time.time()

        # ── 매핑 파일 변경 감지 (2초마다 체크) ───────────────
        if not mapping_mode and now - last_check >= CHECK_INTERVAL:
            last_check = now
            new_mtime  = _get_mtime(ROI_COORDS_PATH)
            if new_mtime and new_mtime != last_mtime:
                # 파일이 바뀌면 호모그래피 리로드 + 상태 머신 재초기화
                last_mtime             = new_mtime
                homography.load()
                new_keys               = list(homography.zones.keys())
                state_machine          = ParkingStateMachine(new_keys)
                ocr_submitted          = {}
                pending_entry          = {}
                empty_snap_initialized = False
                logger.info(f"Mapping reloaded | zones: {new_keys}")

        # ── 카메라 흔들림 감지 + 자동 보정 ───────────────────
        if (not mapping_mode and homography.is_ready()
                and now - last_shake_check >= CAMERA_SHAKE_CHECK_INTERVAL):
            last_shake_check = now
            result = homography.check_and_auto_correct(frame)
            if result == "corrected":
                # 자동 보정 성공
                shake_status_msg  = f"Auto-corrected (x{homography.auto_fix_count})"
                shake_status_time = now
            elif result in ["warning", "marker_lost"]:
                # 심한 흔들림 → 재매핑 권고 메시지
                shake_status_msg  = "WARNING: Camera moved! Press M to re-map"
                shake_status_time = now

        # ── 오래된 스냅샷 파일 정리 ───────────────────────────
        if now - last_snap_cleanup >= SNAPSHOT_CLEANUP_INTERVAL:
            last_snap_cleanup = now
            deleted = _cleanup_snapshots()
            if deleted > 0:
                logger.info(f"[Snapshot] {deleted} old files deleted")

        # ── 주기적 상태 백업 ──────────────────────────────────
        if now - last_state_backup >= STATE_BACKUP_INTERVAL:
            last_state_backup = now
            _backup_state(state_machine)

        # ── 카메라 흐림 감지 (매핑 모드 제외) ────────────────
        if not mapping_mode:
            preprocessor.check_blur(frame)

        # ══════════════════════════════════════════════════
        # 매핑 모드 처리
        # ══════════════════════════════════════════════════
        if mapping_mode:
            # 카메라 화면에 매핑 모드 안내 표시
            cam_vis = mapper.render_camera(frame)
            cv2.putText(cam_vis, "[ MAPPING MODE ]  E: exit",
                        (10, cam_vis.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.imshow(WIN_MAIN, cam_vis)

            # 가상 지도 창이 없으면 생성
            if not virt_win_open:
                cv2.namedWindow(WIN_VIRT)
                cv2.setMouseCallback(WIN_VIRT, mapper.on_mouse)
                virt_win_open = True

            cv2.imshow(WIN_VIRT, mapper.render_virtual())

            # 이름 입력 중이면 30ms 대기, 아니면 1ms 대기
            wait_ms = 30 if mapper.input_mode != "none" else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 255:
                continue

            # Q/ESC: 전체 종료
            if key in [ord('q'), ord('Q'), 27]:
                break

            # E: 매핑 모드 종료 후 일반 모드로 복귀
            if key in [ord('e'), ord('E')]:
                mapping_mode           = False
                virt_win_open          = False
                cv2.destroyWindow(WIN_VIRT)
                # 새로 저장된 매핑 파일 로드
                homography.load()
                new_keys               = list(homography.zones.keys())
                state_machine          = ParkingStateMachine(new_keys)
                ocr_submitted          = {}
                pending_entry          = {}
                last_mtime             = _get_mtime(ROI_COORDS_PATH)
                # 카메라 흔들림 기준점 리셋
                homography.reset_shake_reference(frame)
                shake_status_msg       = "Re-mapping done."
                shake_status_time      = now
                empty_snap_initialized = False
                logger.info(f"Mapping exit | zones: {new_keys}")
                continue

            mapper.handle_key(key, frame)
            continue

        # ══════════════════════════════════════════════════
        # 일반 모드: YOLO 탐지 + 상태 머신 업데이트
        # ══════════════════════════════════════════════════

        # CLAHE 전처리 후 YOLO 탐지
        enhanced         = preprocessor.apply(frame)
        detection_result = detector.detect(enhanced)
        cars             = detection_result["cars"]    # 탐지된 차량 목록
        plates           = detection_result["plates"]  # 탐지된 번호판 목록

        # 카메라 프레임을 가상 평면으로 변환 (호모그래피 적용)
        warped_frame = None
        if homography.is_ready():
            warped_frame = cv2.warpPerspective(
                frame, homography.matrix,
                (VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT)
            )

        # ── 빈 구역 스냅샷 초기화 (최초 1회) ─────────────────
        if not empty_snap_initialized and warped_frame is not None:
            all_done = True
            for zone_name, zone_pts in homography.zones.items():
                zone = state_machine.zones.get(zone_name)
                if zone and zone.empty_snap is None:
                    # 현재 프레임에서 구역 crop 후 빈 스냅샷으로 저장
                    zone_crop = _get_zone_crop(warped_frame, zone_pts)
                    if zone_crop is not None:
                        state_machine.save_empty_snap(zone_name, zone_crop)
                    else:
                        all_done = False
            if all_done:
                empty_snap_initialized = True
                logger.info("[PixelCheck] 빈 구역 스냅샷 초기화 완료")

        # ── 차량 좌표를 가상 평면으로 변환 ───────────────────
        virtual_cars = []
        if homography.is_ready():
            for car in cars:
                # 카메라 좌표계 발바닥 → 가상 평면 좌표로 변환
                vx, vy = homography.camera_to_virtual(
                    (car["foot_x"], car["foot_y"])
                )
                virtual_cars.append({**car, "vx": vx, "vy": vy})

        # ── 2칸 주차 판정 ─────────────────────────────────────
        _check_multi_zone(
            virtual_cars, homography.zones,
            state_machine, send_queue, logger
        )

        # ── 구역별 상태 업데이트 ──────────────────────────────
        for zone_name, zone_pts in homography.zones.items():
            # 이 구역 안에 발바닥이 있는 차량 목록
            cars_in_zone = [
                c for c in virtual_cars
                if point_in_zone((c["vx"], c["vy"]), zone_pts)
            ]

            # 구역 내 첫 번째 차량의 발바닥 좌표 (없으면 None)
            foot = (cars_in_zone[0]["vx"], cars_in_zone[0]["vy"]) \
                   if cars_in_zone else None

            # 번호판 탐지 여부 및 bbox 확인
            plate_visible = False
            plate_bbox    = None
            if cars_in_zone:
                plate_bbox    = detector.find_plate_for_car(
                    cars_in_zone[0], plates
                )
                plate_visible = plate_bbox is not None

            # 가상 평면에서 구역 crop 이미지 추출 (픽셀 비교용)
            zone_crop = _get_zone_crop(warped_frame, zone_pts)

            # 상태 머신 업데이트 → 이벤트 발생 시 딕셔너리 반환
            event = state_machine.update(
                zone_name=zone_name,
                virtual_foot=foot,
                all_cars_in_zone=cars_in_zone,
                plate_visible=plate_visible,
                zone_crop=zone_crop,
            )

            if event:
                logger.info(f"[EVENT] {event}")

                # ── 입차 이벤트 처리 ──────────────────────────
                if event["type"] == "entry":
                    # 입차 시점 스냅샷 저장
                    snap_path = _save_snapshot(
                        frame, zone_name, event["timestamp"]
                    )
                    event["car_image"] = snap_path

                    if not ocr_submitted.get(zone_name, False):
                        task = OcrTask(
                            zone_name   = zone_name,
                            snapshot    = frame.copy(),
                            car_bbox    = cars_in_zone[0] if cars_in_zone else None,
                            plate_bbox  = plate_bbox,
                            entry_event = event,
                        )

                        if not cars_in_zone:
                            # 차량 bbox가 없으면 null로 즉시 전송
                            event["plate"]        = None
                            event["plate_status"] = PlateStatus.NULL.value
                            logger.info(f"[ENTRY] {zone_name} 차량 없음 → null 전송")
                            send_queue.put_nowait((PRIORITY_ENTRY, event))
                        else:
                            try:
                                # OCR 큐에 등록 (OCR Worker가 처리 후 전송)
                                ocr_queue.put_nowait(task)
                                ocr_submitted[zone_name] = True
                                pending_entry[zone_name] = event
                                logger.info(
                                    f"[ENTRY] {zone_name} OCR Queue 제출 "
                                    f"(대기: {ocr_queue.qsize()})"
                                )
                            except queue.Full:
                                # OCR 큐가 가득 찬 경우 null로 바로 전송
                                logger.warning(
                                    f"[ENTRY] {zone_name} Queue 가득참 → null"
                                )
                                event["plate"]        = None
                                event["plate_status"] = PlateStatus.NULL.value
                                send_queue.put_nowait((PRIORITY_ENTRY, event))

                # ── 출차 이벤트 처리 ──────────────────────────
                elif event["type"] == "exit":
                    # 출차 시점 스냅샷 저장
                    exit_snap_path = _save_snapshot(
                        frame, f"{zone_name}_exit", event["timestamp"]
                    )
                    event["exit_image"] = exit_snap_path

                    # OCR 완료 전 출차된 경우: 미전송 입차 이벤트를 null로 먼저 전송
                    pending = pending_entry.pop(zone_name, None)
                    if pending:
                        logger.warning(
                            f"[EXIT] {zone_name} OCR 완료 전 출차 "
                            f"→ entry null 전송"
                        )
                        pending["plate"]        = None
                        pending["plate_status"] = PlateStatus.NULL.value
                        send_queue.put_nowait((PRIORITY_ENTRY, pending))

                    # OCR 제출 상태 초기화
                    ocr_submitted.pop(zone_name, None)

                    # 출차 이벤트를 최우선순위 1로 전송
                    try:
                        send_queue.put_nowait((PRIORITY_EXIT, event))
                    except queue.Full:
                        logger.warning(f"[EXIT] send_queue 가득참")

                    logger.info(f"[EXIT] {zone_name} plate={event['plate']}")

            # ── 번호판 재인식 (주기적 또는 이중주차 의심 시) ──
            if state_machine.needs_recheck(zone_name) and cars_in_zone:
                cur = state_machine.zones[zone_name]
                if not ocr_submitted.get(zone_name, False):
                    if not plate_visible:
                        # 번호판 안 보이면 재인식 패스
                        state_machine.mark_rechecked(zone_name)
                    else:
                        # 번호판 재인식 시도
                        new_plate = ocr_reader.recheck(
                            frame=frame,
                            bbox=cars_in_zone[0],
                            zone_name=zone_name,
                            prev_plate=cur.plate,
                        )
                        state_machine.mark_rechecked(zone_name)

                        if new_plate:
                            logger.info(
                                f"[RECHECK] {zone_name}: "
                                f"{cur.plate} -> {new_plate}"
                            )
                            try:
                                if cur.plate is None:
                                    # 기존 번호판이 없으면 → plate_update 이벤트
                                    state_machine.set_plate(zone_name, new_plate)
                                    send_queue.put_nowait((PRIORITY_UPDATE, {
                                        "type":         "plate_update",
                                        "zone":         zone_name,
                                        "plate":        new_plate,
                                        "plate_status": "confirmed",
                                        "entry_time":   cur.entry_time,
                                        "park_status":  cur.park_status.value,
                                        "linked_zone":  cur.linked_zone,
                                        "timestamp":    time.time(),
                                    }))
                                    logger.info(
                                        f"[PLATE UPDATE] {zone_name} "
                                        f"null -> {new_plate}"
                                    )
                                else:
                                    # 기존 번호판과 다른 차 → exit 후 plate_changed
                                    # 출차 먼저 전송 (우선순위 1)
                                    send_queue.put_nowait((PRIORITY_EXIT, {
                                        "type":         "exit",
                                        "zone":         zone_name,
                                        "plate":        cur.plate,
                                        "plate_status": cur.plate_status.value,
                                        "entry_time":   cur.entry_time,
                                        "park_status":  cur.park_status.value,
                                        "linked_zone":  cur.linked_zone,
                                        "timestamp":    time.time(),
                                    }))
                                    # 새 번호판으로 업데이트 (우선순위 3)
                                    state_machine.set_plate(zone_name, new_plate)
                                    send_queue.put_nowait((PRIORITY_UPDATE, {
                                        "type":         "plate_changed",
                                        "zone":         zone_name,
                                        "plate":        new_plate,
                                        "plate_status": "confirmed",
                                        "entry_time":   cur.entry_time,
                                        "park_status":  cur.park_status.value,
                                        "linked_zone":  cur.linked_zone,
                                        "timestamp":    time.time(),
                                    }))
                            except queue.Full:
                                pass

        # ── FPS 계산 ──────────────────────────────────────
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── 시각화 ────────────────────────────────────────
        vis_frame = visualizer.draw_frame(
            frame=frame,
            cars=cars,
            plates=plates,
            zone_statuses=state_machine.get_all_status(),
            homography_transformer=homography,
            fps=fps,
            state_machine=state_machine,
        )

        # 매핑 미완료 시 안내 메시지
        if not homography.is_ready():
            cv2.putText(vis_frame,
                        "No mapping  |  Press M to enter mapping mode",
                        (10, vis_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # OCR/Send 큐 크기 표시 (디버깅용)
        q_ocr  = ocr_queue.qsize()
        q_send = send_queue.qsize()
        if q_ocr > 0 or q_send > 0:
            cv2.putText(vis_frame,
                        f"OCR:{q_ocr} Send:{q_send}",
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 255), 1)

        # 흔들림 상태 메시지 표시 (3초간)
        if shake_status_msg and now - shake_status_time < STATUS_DISPLAY_SEC:
            is_warn = "WARNING" in shake_status_msg
            color   = (0, 60, 255) if is_warn else (0, 200, 80)
            cv2.putText(vis_frame, shake_status_msg,
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        elif now - shake_status_time >= STATUS_DISPLAY_SEC:
            shake_status_msg = ""

        # 카메라 오염 경고 표시
        if preprocessor.camera_blurry:
            warn_txt = "! CAM DIRTY"
            (tw, th), _ = cv2.getTextSize(
                warn_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            # 우상단에 빨간 배경 경고 박스
            wx = vis_frame.shape[1] - tw - 10
            wy = 45
            cv2.rectangle(vis_frame,
                          (wx - 4, wy - th - 4),
                          (wx + tw + 4, wy + 4),
                          (0, 0, 180), -1)
            cv2.putText(vis_frame, warn_txt,
                        (wx, wy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)

        cv2.imshow(WIN_MAIN, vis_frame)

        # ── 키 입력 처리 ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            # Q/ESC: 종료
            logger.info("Quit")
            break
        elif key in [ord('m'), ord('M')]:
            # M: 매핑 모드 진입
            logger.info("Enter mapping mode")
            mapping_mode = True
            mapper.load_existing()

    # ── 종료 처리 ──────────────────────────────────────────
    logger.info("[Workers] 종료 대기 중...")
    stop_event.set()  # 모든 워커 스레드에 종료 신호
    for t in ocr_workers:
        t.join(timeout=5.0)
    send_thread.join(timeout=5.0)

    # 종료 전 마지막 상태 백업
    _backup_state(state_machine)
    cap.release()
    cv2.destroyAllWindows()
    logger.info("System stopped")


# ── 상태 백업 ─────────────────────────────────────────────

def _backup_state(state_machine):
    """현재 구역 상태를 JSON 파일에 저장 (전원 차단 복구용)."""
    try:
        data = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "zones":    state_machine.to_dict(),
        }
        with open(STATE_BACKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Backup] Save failed: {e}")


def _restore_state(state_machine):
    """백업 파일에서 상태 복구. 60분 이상 지난 백업은 무시."""
    if not os.path.exists(STATE_BACKUP_PATH):
        return
    try:
        with open(STATE_BACKUP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved_at    = datetime.strptime(
            data.get("saved_at", ""), "%Y-%m-%d %H:%M:%S"
        )
        age_minutes = (datetime.now() - saved_at).total_seconds() / 60
        if age_minutes > 60:
            # 60분 이상 지난 백업은 무시 (주차 상태가 이미 변했을 가능성)
            print(f"[Restore] Backup too old ({age_minutes:.0f}min) - skip")
            return
        zones_data = data.get("zones", {})
        if not zones_data:
            return
        state_machine.from_dict(zones_data)
        print(f"[Restore] State restored ({age_minutes:.1f}min ago)")
    except Exception as e:
        print(f"[Restore] Failed: {e}")


# ── 2칸 주차 판정 ─────────────────────────────────────────

def _check_multi_zone(virtual_cars, zones, state_machine,
                      send_queue, logger):
    """
    2칸 주차 판정 로직.

    각 구역에서 가장 많이 겹친 차량을 선택하고,
    한 차량이 2개 이상 구역에 걸쳐있으면 2칸 주차로 처리.
    겹침 비율이 더 높은 차가 구역 소유권을 가져감.
    """
    # ── 1단계: 구역별로 가장 많이 겹친 차량 결정 ────────────
    # zone_best_car[zone_name] = (car, overlap_ratio)
    zone_best_car = {}
    for zone_name, zone_pts in zones.items():
        best_car   = None
        best_ratio = 0.0
        for car in virtual_cars:
            # 발바닥이 구역 안에 없으면 스킵
            if not point_in_zone((car["vx"], car["vy"]), zone_pts):
                continue
            # bbox 기준 겹침 비율 계산
            ratio = _calc_bbox_zone_overlap(car, zone_pts)
            if ratio > best_ratio:
                best_ratio = ratio
                best_car   = car
        # 겹침 비율이 임계값 이상인 경우만 등록
        if best_car is not None and best_ratio >= MULTI_ZONE_OVERLAP_RATIO:
            zone_best_car[zone_name] = (best_car, best_ratio)

    # ── 2단계: 차량별로 점유한 구역 목록 수집 ───────────────
    # car_zones[car_id] = {"car": ..., "zones": [(zone_name, ratio), ...]}
    car_zones = {}
    for zone_name, (car, ratio) in zone_best_car.items():
        car_id = id(car)
        if car_id not in car_zones:
            car_zones[car_id] = {"car": car, "zones": []}
        car_zones[car_id]["zones"].append((zone_name, ratio))

    # ── 3단계: 2개 이상 구역에 걸친 차량 → 2칸 주차 처리 ────
    for car_id, info in car_zones.items():
        car           = info["car"]
        car_zone_list = info["zones"]

        # 1개 구역만 점유 중이면 2칸 주차 아님
        if len(car_zone_list) < 2:
            continue

        # 겹침 비율 내림차순 정렬 후 상위 2개 구역 선택
        car_zone_list.sort(key=lambda x: x[1], reverse=True)
        za = car_zone_list[0][0]
        zb = car_zone_list[1][0]

        state_a = state_machine.zones.get(za)
        state_b = state_machine.zones.get(zb)

        if state_a is None or state_b is None:
            continue

        # 이미 같은 차량으로 두 구역이 연결된 경우 중복 처리 방지
        if (state_a.status == ZoneStatus.OCCUPIED and
                state_b.status == ZoneStatus.OCCUPIED and
                state_a.linked_zone == zb and
                state_b.linked_zone == za):
            continue

        # 두 구역 모두 이미 점유 중이면 스킵
        if (state_a.status == ZoneStatus.OCCUPIED and
                state_b.status == ZoneStatus.OCCUPIED):
            continue

        logger.info(f"[MULTI-ZONE] CONFIRMED: {za}({car_zone_list[0][1]:.2f}) "
                    f"and {zb}({car_zone_list[1][1]:.2f})")

        # 두 구역을 2칸 주차로 설정
        state_machine.set_multi_zone(za, zb, None)

        # 두 구역 모두 입차 이벤트 전송
        for zn in [za, zb]:
            z = state_machine.zones.get(zn)
            if z is None:
                continue
            try:
                send_queue.put_nowait((PRIORITY_ENTRY, {
                    "type":         "entry",
                    "zone":         zn,
                    "plate":        None,
                    "plate_status": "null",
                    "entry_time":   z.entry_time,
                    "park_status":  "multi_zone",
                    "linked_zone":  z.linked_zone,
                    "timestamp":    time.time(),
                }))
            except queue.Full:
                pass


def _calc_bbox_zone_overlap(car, zone_pts) -> float:
    """
    차량 발바닥 주변 추정 bbox와 구역 폴리곤의 겹침 비율 계산.
    가상 평면 좌표 기준으로 계산.
    """
    try:
        vx, vy    = car["vx"], car["vy"]
        # 카메라 bbox의 30% 크기로 가상 평면 추정 bbox 생성
        bw        = (car["x2"] - car["x1"]) * 0.3
        bh        = (car["y2"] - car["y1"]) * 0.3
        car_poly  = np.float32([
            [vx-bw/2, vy-bh], [vx+bw/2, vy-bh],
            [vx+bw/2, vy   ], [vx-bw/2, vy   ],
        ])
        zone_poly = np.float32(zone_pts)
        zone_area = cv2.contourArea(zone_poly)
        if zone_area == 0:
            return 0.0
        # 교차 면적 / 구역 면적 = 겹침 비율
        return _polygon_intersection_area(car_poly, zone_poly) / zone_area
    except Exception:
        return 0.0


def _polygon_intersection_area(poly1, poly2) -> float:
    """Sutherland-Hodgman 알고리즘으로 두 폴리곤 교차 면적 계산."""

    def inside(p, a, b):
        # 점 p가 엣지 a→b의 안쪽(왼쪽)에 있는지 판별
        return ((b[0]-a[0])*(p[1]-a[1])) > ((b[1]-a[1])*(p[0]-a[0]))

    def intersect(p1, p2, p3, p4):
        # 두 선분의 교점 계산
        x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
        d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if abs(d) < 1e-10:
            return p1  # 평행이면 첫 점 반환
        t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)) / d
        return (x1+t*(x2-x1), y1+t*(y2-y1))

    def clip(subj, cpoly):
        # Sutherland-Hodgman 클리핑
        out = list(map(tuple, subj))
        for i in range(len(cpoly)):
            if not out:
                return []
            inp = out; out = []
            a = tuple(cpoly[i]); b = tuple(cpoly[(i+1) % len(cpoly)])
            for k in range(len(inp)):
                c = inp[k]; p = inp[k-1]
                if inside(c, a, b):
                    if not inside(p, a, b):
                        out.append(intersect(p, c, a, b))
                    out.append(c)
                elif inside(p, a, b):
                    out.append(intersect(p, c, a, b))
        return out

    clipped = clip(poly1, poly2)
    if len(clipped) < 3:
        return 0.0

    # 신발끈 공식으로 면적 계산
    n = len(clipped)
    area = 0.0
    for i in range(n):
        j = (i+1) % n
        area += clipped[i][0]*clipped[j][1] - clipped[j][0]*clipped[i][1]
    return abs(area) / 2.0


# ── 유틸리티 함수들 ───────────────────────────────────────

def _get_zone_crop(warped_frame, zone_pts):
    """가상 평면 프레임에서 구역 폴리곤의 bounding box 영역 crop."""
    if warped_frame is None or not zone_pts:
        return None
    try:
        pts        = np.array(zone_pts, dtype=np.int32)
        # 폴리곤을 감싸는 사각형 bounding box 계산
        x, y, w, h = cv2.boundingRect(pts)
        ih, iw     = warped_frame.shape[:2]
        # 프레임 경계 내로 클리핑
        x  = max(0, x);    y  = max(0, y)
        x2 = min(iw, x+w); y2 = min(ih, y+h)
        if x2 <= x or y2 <= y:
            return None
        return warped_frame[y:y2, x:x2].copy()
    except Exception:
        return None


def _cleanup_snapshots() -> int:
    """SNAPSHOT_MAX_AGE_HOURS 시간 이상 된 스냅샷 파일 삭제."""
    deleted = 0
    cutoff  = datetime.now() - timedelta(hours=SNAPSHOT_MAX_AGE_HOURS)
    try:
        for fname in os.listdir(SNAPSHOT_DIR):
            if not fname.endswith(".jpg"):
                continue
            fpath = os.path.join(SNAPSHOT_DIR, fname)
            try:
                # 파일 수정 시각이 기준 시각보다 오래됐으면 삭제
                if datetime.fromtimestamp(
                        os.path.getmtime(fpath)) < cutoff:
                    os.remove(fpath)
                    deleted += 1
            except Exception:
                continue
    except Exception:
        pass
    return deleted


def _save_snapshot(frame, zone_name, timestamp) -> str | None:
    """입출차 시점 프레임을 JPG로 저장하고 경로 반환."""
    try:
        dt   = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"{zone_name}_{dt}.jpg")
        cv2.imwrite(path, frame)
        return path
    except Exception:
        return None


def _get_mtime(path):
    """파일의 마지막 수정 시각 반환. 파일 없으면 None."""
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    main()