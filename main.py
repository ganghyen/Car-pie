# ============================================================
# [메인] 스마트 주차 관리 시스템
#
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

WIN_MAIN = "Smart Parking  |  M: Mapping   Q: Quit"
WIN_VIRT = "Virtual Map  |  Click 4pts  S: Save  X: Delete  C: Cancel  E: Exit"

# ── 우선순위 상수 ─────────────────────────────────────────
PRIORITY_EXIT   = 1  # 출차 최우선
PRIORITY_ENTRY  = 2  # 입차
PRIORITY_UPDATE = 3  # 번호판 업데이트


# ── OCR 작업 단위 ─────────────────────────────────────────
class OcrTask:
    def __init__(self, zone_name, snapshot, car_bbox,
                 plate_bbox, entry_event):
        self.zone_name   = zone_name
        self.snapshot    = snapshot
        self.car_bbox    = car_bbox
        self.plate_bbox  = plate_bbox
        self.entry_event = entry_event
        self.queued_at   = time.time()


# ── OCR Worker ────────────────────────────────────────────
def ocr_worker(ocr_queue: queue.Queue,
               send_queue: queue.PriorityQueue,
               ocr_reader: PlateReader,
               state_machine: ParkingStateMachine,
               stop_event: threading.Event):
    logger.info("[OCR Worker] 시작")

    while not stop_event.is_set():
        try:
            task: OcrTask = ocr_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            zone_name = task.zone_name
            plate = ocr_reader.vote_from_snapshot(
                snapshot_frame=task.snapshot,
                bbox=task.car_bbox,
                zone_name=zone_name,
                plate_bbox=task.plate_bbox,
            )

            state_machine.set_plate(zone_name, plate)
            zone = state_machine.zones.get(zone_name)
            ps   = zone.plate_status.value if zone else "null"

            logger.info(f"[OCR Worker] {zone_name} 완료: {plate} ({ps})")

            entry_event                 = task.entry_event
            entry_event["plate"]        = zone.plate if zone else plate
            entry_event["plate_status"] = ps

            # ★ 입차는 우선순위 2로 전송
            send_queue.put_nowait((PRIORITY_ENTRY, entry_event))

            if zone and zone.plate_status == PlateStatus.UNREADABLE:
                logger.warning(f"[UNREADABLE] {zone_name} 번호판 인식 불가")

        except Exception as e:
            logger.error(f"[OCR Worker] {task.zone_name} 오류: {e}")
        finally:
            ocr_queue.task_done()

    logger.info("[OCR Worker] 종료")


# ── Send Worker ───────────────────────────────────────────
def send_worker(send_queue: queue.PriorityQueue,
                sender: EventSender,
                stop_event: threading.Event):
    logger.info("[Send Worker] 시작")

    while not stop_event.is_set():
        try:
            # PriorityQueue에서 꺼낼 때 (우선순위, 이벤트) 형태
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

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_BACKUP_PATH), exist_ok=True)

    homography   = HomographyTransformer()
    preprocessor = Preprocessor()
    detector     = VehicleDetector()
    ocr_reader   = PlateReader()
    sender       = EventSender()
    visualizer   = Visualizer()
    mapper       = ROIMapper()

    homography.load()
    mapper.load_existing()

    if not homography.is_ready():
        logger.warning("No mapping - Press M to enter mapping mode")
    else:
        logger.info(f"Mapping loaded | zones: {list(homography.zones.keys())}")

    zone_keys     = list(homography.zones.keys()) if homography.zones else []
    state_machine = ParkingStateMachine(zone_keys)

    _restore_state(state_machine)

    last_mtime     = _get_mtime(ROI_COORDS_PATH)
    CHECK_INTERVAL = 2.0
    last_check     = time.time()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -3)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 120)
    cap.set(cv2.CAP_PROP_CONTRAST, 150)
    cap.set(cv2.CAP_PROP_SHARPNESS, 200)

    time.sleep(5.0)
    logger.info("[Camera] C920 자동 포커스 설정 완료")

    if not cap.isOpened():
        logger.error(f"Camera {CAMERA_INDEX} open failed")
        sys.exit(1)

    ocr_queue  = queue.Queue(maxsize=20)
    # ★ PriorityQueue로 변경 (출차>입차>업데이트 순서 보장)
    send_queue = queue.PriorityQueue(maxsize=50)
    stop_event = threading.Event()

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

    send_thread = threading.Thread(
        target=send_worker,
        args=(send_queue, sender, stop_event),
        daemon=True,
        name="Send-Worker"
    )
    send_thread.start()

    logger.info(f"[Workers] OCR x{OCR_MAX_THREADS}, Send x1 시작")

    ocr_submitted: dict[str, bool] = {}
    pending_entry: dict[str, dict] = {}

    last_shake_check  = time.time()
    last_snap_cleanup = time.time()
    last_state_backup = time.time()

    shake_status_msg   = ""
    shake_status_time  = 0.0
    STATUS_DISPLAY_SEC = 3.0

    empty_snap_initialized = False
    mapping_mode  = False
    virt_win_open = False

    cv2.namedWindow(WIN_MAIN)
    prev_time = time.time()
    logger.info("Camera started | M: mapping  Q/ESC: quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        now = time.time()

        if not mapping_mode and now - last_check >= CHECK_INTERVAL:
            last_check = now
            new_mtime  = _get_mtime(ROI_COORDS_PATH)
            if new_mtime and new_mtime != last_mtime:
                last_mtime             = new_mtime
                homography.load()
                new_keys               = list(homography.zones.keys())
                state_machine          = ParkingStateMachine(new_keys)
                ocr_submitted          = {}
                pending_entry          = {}
                empty_snap_initialized = False
                logger.info(f"Mapping reloaded | zones: {new_keys}")

        if (not mapping_mode and homography.is_ready()
                and now - last_shake_check >= CAMERA_SHAKE_CHECK_INTERVAL):
            last_shake_check = now
            result = homography.check_and_auto_correct(frame)
            if result == "corrected":
                shake_status_msg  = f"Auto-corrected (x{homography.auto_fix_count})"
                shake_status_time = now
            elif result in ["warning", "marker_lost"]:
                shake_status_msg  = "WARNING: Camera moved! Press M to re-map"
                shake_status_time = now

        if now - last_snap_cleanup >= SNAPSHOT_CLEANUP_INTERVAL:
            last_snap_cleanup = now
            deleted = _cleanup_snapshots()
            if deleted > 0:
                logger.info(f"[Snapshot] {deleted} old files deleted")

        if now - last_state_backup >= STATE_BACKUP_INTERVAL:
            last_state_backup = now
            _backup_state(state_machine)

        if not mapping_mode:
            preprocessor.check_blur(frame)

        if mapping_mode:
            cam_vis = mapper.render_camera(frame)
            cv2.putText(cam_vis, "[ MAPPING MODE ]  E: exit",
                        (10, cam_vis.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.imshow(WIN_MAIN, cam_vis)

            if not virt_win_open:
                cv2.namedWindow(WIN_VIRT)
                cv2.setMouseCallback(WIN_VIRT, mapper.on_mouse)
                virt_win_open = True

            cv2.imshow(WIN_VIRT, mapper.render_virtual())

            wait_ms = 30 if mapper.input_mode != "none" else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 255:
                continue

            if key in [ord('q'), ord('Q'), 27]:
                break

            if key in [ord('e'), ord('E')]:
                mapping_mode           = False
                virt_win_open          = False
                cv2.destroyWindow(WIN_VIRT)
                homography.load()
                new_keys               = list(homography.zones.keys())
                state_machine          = ParkingStateMachine(new_keys)
                ocr_submitted          = {}
                pending_entry          = {}
                last_mtime             = _get_mtime(ROI_COORDS_PATH)
                homography.reset_shake_reference(frame)
                shake_status_msg       = "Re-mapping done."
                shake_status_time      = now
                empty_snap_initialized = False
                logger.info(f"Mapping exit | zones: {new_keys}")
                continue

            mapper.handle_key(key, frame)
            continue

        enhanced         = preprocessor.apply(frame)
        detection_result = detector.detect(enhanced)
        cars             = detection_result["cars"]
        plates           = detection_result["plates"]

        warped_frame = None
        if homography.is_ready():
            warped_frame = cv2.warpPerspective(
                frame, homography.matrix,
                (VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT)
            )

        if not empty_snap_initialized and warped_frame is not None:
            all_done = True
            for zone_name, zone_pts in homography.zones.items():
                zone = state_machine.zones.get(zone_name)
                if zone and zone.empty_snap is None:
                    zone_crop = _get_zone_crop(warped_frame, zone_pts)
                    if zone_crop is not None:
                        state_machine.save_empty_snap(zone_name, zone_crop)
                    else:
                        all_done = False
            if all_done:
                empty_snap_initialized = True
                logger.info("[PixelCheck] 빈 구역 스냅샷 초기화 완료")

        virtual_cars = []
        if homography.is_ready():
            for car in cars:
                vx, vy = homography.camera_to_virtual(
                    (car["foot_x"], car["foot_y"])
                )
                virtual_cars.append({**car, "vx": vx, "vy": vy})

        _check_multi_zone(
            virtual_cars, homography.zones,
            state_machine, send_queue, logger
        )

        for zone_name, zone_pts in homography.zones.items():
            cars_in_zone = [
                c for c in virtual_cars
                if point_in_zone((c["vx"], c["vy"]), zone_pts)
            ]

            foot = (cars_in_zone[0]["vx"], cars_in_zone[0]["vy"]) \
                   if cars_in_zone else None

            plate_visible = False
            plate_bbox    = None
            if cars_in_zone:
                plate_bbox    = detector.find_plate_for_car(
                    cars_in_zone[0], plates
                )
                plate_visible = plate_bbox is not None

            zone_crop = _get_zone_crop(warped_frame, zone_pts)

            event = state_machine.update(
                zone_name=zone_name,
                virtual_foot=foot,
                all_cars_in_zone=cars_in_zone,
                plate_visible=plate_visible,
                zone_crop=zone_crop,
            )

            if event:
                logger.info(f"[EVENT] {event}")

                if event["type"] == "entry":
                    snap_path = _save_snapshot(
                        frame, zone_name, event["timestamp"]
                    )
                    event["car_image"] = snap_path

                    # ★ 번호판 안 보여도 null 즉시 전송 안함
                    # → 항상 OCR Queue에 넣어서 처리 후 전송
                    if not ocr_submitted.get(zone_name, False):
                        task = OcrTask(
                            zone_name   = zone_name,
                            snapshot    = frame.copy(),
                            car_bbox    = cars_in_zone[0] if cars_in_zone else None,
                            plate_bbox  = plate_bbox,
                            entry_event = event,
                        )

                        # 차량 없거나 번호판 안 보이면 null로 바로 OCR Worker 통해 전송
                        if not cars_in_zone:
                            event["plate"]        = None
                            event["plate_status"] = PlateStatus.NULL.value
                            logger.info(f"[ENTRY] {zone_name} 차량 없음 → null 전송")
                            send_queue.put_nowait((PRIORITY_ENTRY, event))
                        else:
                            try:
                                ocr_queue.put_nowait(task)
                                ocr_submitted[zone_name] = True
                                pending_entry[zone_name] = event
                                logger.info(
                                    f"[ENTRY] {zone_name} OCR Queue 제출 "
                                    f"(대기: {ocr_queue.qsize()})"
                                )
                            except queue.Full:
                                logger.warning(
                                    f"[ENTRY] {zone_name} Queue 가득참 → null"
                                )
                                event["plate"]        = None
                                event["plate_status"] = PlateStatus.NULL.value
                                send_queue.put_nowait((PRIORITY_ENTRY, event))

                elif event["type"] == "exit":
                    exit_snap_path = _save_snapshot(
                        frame, f"{zone_name}_exit", event["timestamp"]
                    )
                    event["exit_image"] = exit_snap_path

                    pending = pending_entry.pop(zone_name, None)
                    if pending:
                        logger.warning(
                            f"[EXIT] {zone_name} OCR 완료 전 출차 "
                            f"→ entry null 전송"
                        )
                        pending["plate"]        = None
                        pending["plate_status"] = PlateStatus.NULL.value
                        send_queue.put_nowait((PRIORITY_ENTRY, pending))

                    ocr_submitted.pop(zone_name, None)

                    # ★ 출차는 최우선순위 1로 전송
                    try:
                        send_queue.put_nowait((PRIORITY_EXIT, event))
                    except queue.Full:
                        logger.warning(f"[EXIT] send_queue 가득참")

                    logger.info(f"[EXIT] {zone_name} plate={event['plate']}")

            if state_machine.needs_recheck(zone_name) and cars_in_zone:
                cur = state_machine.zones[zone_name]
                if not ocr_submitted.get(zone_name, False):
                    if not plate_visible:
                        state_machine.mark_rechecked(zone_name)
                    else:
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
                                    state_machine.set_plate(zone_name, new_plate)
                                    # ★ 업데이트는 우선순위 3
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
                                    # ★ 출차는 최우선순위 1
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

        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        vis_frame = visualizer.draw_frame(
            frame=frame,
            cars=cars,
            plates=plates,
            zone_statuses=state_machine.get_all_status(),
            homography_transformer=homography,
            fps=fps,
            state_machine=state_machine,
        )

        if not homography.is_ready():
            cv2.putText(vis_frame,
                        "No mapping  |  Press M to enter mapping mode",
                        (10, vis_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        q_ocr  = ocr_queue.qsize()
        q_send = send_queue.qsize()
        if q_ocr > 0 or q_send > 0:
            cv2.putText(vis_frame,
                        f"OCR:{q_ocr} Send:{q_send}",
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 255), 1)

        if shake_status_msg and now - shake_status_time < STATUS_DISPLAY_SEC:
            is_warn = "WARNING" in shake_status_msg
            color   = (0, 60, 255) if is_warn else (0, 200, 80)
            cv2.putText(vis_frame, shake_status_msg,
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        elif now - shake_status_time >= STATUS_DISPLAY_SEC:
            shake_status_msg = ""

        if preprocessor.camera_blurry:
            warn_txt = "! CAM DIRTY"
            (tw, th), _ = cv2.getTextSize(
                warn_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
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

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            logger.info("Quit")
            break
        elif key in [ord('m'), ord('M')]:
            logger.info("Enter mapping mode")
            mapping_mode = True
            mapper.load_existing()

    logger.info("[Workers] 종료 대기 중...")
    stop_event.set()
    for t in ocr_workers:
        t.join(timeout=5.0)
    send_thread.join(timeout=5.0)

    _backup_state(state_machine)
    cap.release()
    cv2.destroyAllWindows()
    logger.info("System stopped")


# ── 상태 백업 / 복구 ──────────────────────────────────────

def _backup_state(state_machine):
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
            print(f"[Restore] Backup too old ({age_minutes:.0f}min) - skip")
            return
        zones_data = data.get("zones", {})
        if not zones_data:
            return
        state_machine.from_dict(zones_data)
        print(f"[Restore] State restored ({age_minutes:.1f}min ago)")
    except Exception as e:
        print(f"[Restore] Failed: {e}")


def _check_multi_zone(virtual_cars, zones, state_machine,
                      send_queue, logger):
    """
    2칸 주차 판정 + 겹침 구역 소유권 결정

    차 2대가 A-1, A-2, A-3에 각각 주차할 때:
    - 차1이 A-1, A-2에 걸침
    - 차2가 A-2, A-3에 걸침
    - A-2는 겹침 비율이 더 높은 차가 가져감
    """
    zone_names = list(zones.keys())

    # ── 1. 각 구역별로 가장 많이 걸친 차 결정 ──────────────
    # zone_best_car[zone_name] = (car, overlap_ratio)
    zone_best_car = {}
    for zone_name, zone_pts in zones.items():
        best_car   = None
        best_ratio = 0.0
        for car in virtual_cars:
            if not point_in_zone((car["vx"], car["vy"]), zone_pts):
                continue
            ratio = _calc_bbox_zone_overlap(car, zone_pts)
            if ratio > best_ratio:
                best_ratio = ratio
                best_car   = car
        if best_car is not None and best_ratio >= MULTI_ZONE_OVERLAP_RATIO:
            zone_best_car[zone_name] = (best_car, best_ratio)

    # ── 2. 차량별로 어느 구역에 걸쳐있는지 계산 ──────────
    # car_zones[car_id] = [(zone_name, overlap_ratio), ...]
    car_zones = {}
    for zone_name, (car, ratio) in zone_best_car.items():
        car_id = id(car)
        if car_id not in car_zones:
            car_zones[car_id] = {"car": car, "zones": []}
        car_zones[car_id]["zones"].append((zone_name, ratio))

    # ── 3. 2개 이상 구역에 걸친 차 → 2칸 주차 처리 ───────
    for car_id, info in car_zones.items():
        car        = info["car"]
        car_zone_list = info["zones"]

        if len(car_zone_list) < 2:
            continue

        # 가장 많이 걸친 2개 구역 선택
        car_zone_list.sort(key=lambda x: x[1], reverse=True)
        za = car_zone_list[0][0]
        zb = car_zone_list[1][0]

        state_a = state_machine.zones.get(za)
        state_b = state_machine.zones.get(zb)

        if state_a is None or state_b is None:
            continue

        # 이미 같은 차량으로 linked 된 경우 스킵
        if (state_a.status == ZoneStatus.OCCUPIED and
                state_b.status == ZoneStatus.OCCUPIED and
                state_a.linked_zone == zb and
                state_b.linked_zone == za):
            continue

        # 둘 다 비어있거나 한쪽만 비어있을 때만 처리
        if (state_a.status == ZoneStatus.OCCUPIED and
                state_b.status == ZoneStatus.OCCUPIED):
            continue

        logger.info(f"[MULTI-ZONE] CONFIRMED: {za}({car_zone_list[0][1]:.2f}) "
                    f"and {zb}({car_zone_list[1][1]:.2f})")

        state_machine.set_multi_zone(za, zb, None)

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
    try:
        vx, vy    = car["vx"], car["vy"]
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
        return _polygon_intersection_area(car_poly, zone_poly) / zone_area
    except Exception:
        return 0.0


def _polygon_intersection_area(poly1, poly2) -> float:
    def inside(p, a, b):
        return ((b[0]-a[0])*(p[1]-a[1])) > ((b[1]-a[1])*(p[0]-a[0]))

    def intersect(p1, p2, p3, p4):
        x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
        d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if abs(d) < 1e-10:
            return p1
        t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)) / d
        return (x1+t*(x2-x1), y1+t*(y2-y1))

    def clip(subj, cpoly):
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
    n = len(clipped)
    area = 0.0
    for i in range(n):
        j = (i+1) % n
        area += clipped[i][0]*clipped[j][1] - clipped[j][0]*clipped[i][1]
    return abs(area) / 2.0


def _get_zone_crop(warped_frame, zone_pts):
    if warped_frame is None or not zone_pts:
        return None
    try:
        pts        = np.array(zone_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        ih, iw     = warped_frame.shape[:2]
        x  = max(0, x);    y  = max(0, y)
        x2 = min(iw, x+w); y2 = min(ih, y+h)
        if x2 <= x or y2 <= y:
            return None
        return warped_frame[y:y2, x:x2].copy()
    except Exception:
        return None


def _cleanup_snapshots() -> int:
    deleted = 0
    cutoff  = datetime.now() - timedelta(hours=SNAPSHOT_MAX_AGE_HOURS)
    try:
        for fname in os.listdir(SNAPSHOT_DIR):
            if not fname.endswith(".jpg"):
                continue
            fpath = os.path.join(SNAPSHOT_DIR, fname)
            try:
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
    try:
        dt   = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"{zone_name}_{dt}.jpg")
        cv2.imwrite(path, frame)
        return path
    except Exception:
        return None


def _get_mtime(path):
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    main()
