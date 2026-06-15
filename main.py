# ============================================================
# [메인] 스마트 주차 관리 시스템 9 - 헤드리스(GUI 없음)
# ============================================================

import cv2
import time
import sys
import os
import json
import base64
import threading
import queue
import numpy as np
import itertools
import requests
from datetime import datetime, timedelta
from config.settings import STILL_SECONDS_REQUIRED

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
    STATUS_REPORT_INTERVAL, STATUS_REPORT_URL,
)
from mapping.homography import HomographyTransformer
from core.preprocessor import Preprocessor
from core.detector import VehicleDetector
from state.overlap import point_in_zone, bbox_overlap_ratio
from state.zone_state import (
    ParkingStateMachine, ZoneStatus, PlateStatus, ParkStatus
)
from ocr.reader import PlateReader, PLATE_UNREADABLE
from comm.sender import EventSender
from utils.logger import get_logger

logger = get_logger("parking")

PRIORITY_EXIT   = 1
PRIORITY_ENTRY  = 2
PRIORITY_UPDATE = 3
_counter = itertools.count()

# OCR 큐 우선순위
OCR_PRIORITY_ENTRY   = 0
OCR_PRIORITY_RECHECK = 1

# 구역별 중앙점 감지 확인 로그 주기 (초)
DETECT_LOG_INTERVAL = 5.0


def _frame_to_base64(frame) -> str | None:
    try:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception:
        return None


def _ensure_image_base64(event: dict, frame, zone_name: str, reason: str):
    if event.get("image_base64") or frame is None:
        return
    image_b64 = _frame_to_base64(frame)
    if image_b64:
        event["image_base64"] = image_b64


class OcrTask:
    def __init__(self, zone_name, snapshot, car_bbox,
                 plate_bbox, entry_event, is_recheck=False):
        self.zone_name   = zone_name
        self.snapshot    = snapshot
        self.car_bbox    = car_bbox
        self.plate_bbox  = plate_bbox
        self.entry_event = entry_event
        self.queued_at   = time.time()
        self.is_recheck  = is_recheck


def ocr_worker(ocr_queue: queue.PriorityQueue,
               send_queue: queue.PriorityQueue,
               ocr_reader: PlateReader,
               state_machine: ParkingStateMachine,
               stop_event: threading.Event,
               ocr_submitted: dict):

    while not stop_event.is_set():
        try:
            _, _, task = ocr_queue.get(timeout=1.0)
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

            if task.is_recheck:
                zone = state_machine.zones.get(zone_name)
                if zone is None or zone.status == ZoneStatus.EMPTY:
                    ocr_submitted.pop(zone_name, None)
                    continue

                prev_plate = task.entry_event.get("prev_plate")
                if plate and plate != PLATE_UNREADABLE and plate != prev_plate:
                    try:
                        if prev_plate is None:
                            state_machine.set_plate(zone_name, plate)
                            send_queue.put_nowait((PRIORITY_UPDATE, next(_counter), {
                                "type":         "plate_update",
                                "zone":         zone_name,
                                "plate":        plate,
                                "plate_status": "confirmed",
                                "entry_time":   zone.entry_time,
                                "park_status":  zone.park_status.value,
                                "linked_zone":  zone.linked_zone,
                                "timestamp":    time.time(),
                            }))
                        else:
                            send_queue.put_nowait((PRIORITY_EXIT, next(_counter), {
                                "type":         "exit",
                                "zone":         zone_name,
                                "plate":        prev_plate,
                                "plate_status": zone.plate_status.value,
                                "entry_time":   zone.entry_time,
                                "park_status":  zone.park_status.value,
                                "linked_zone":  zone.linked_zone,
                                "timestamp":    time.time(),
                            }))
                            state_machine.set_plate(zone_name, plate)
                            send_queue.put_nowait((PRIORITY_UPDATE, next(_counter), {
                                "type":         "plate_changed",
                                "zone":         zone_name,
                                "plate":        plate,
                                "plate_status": "confirmed",
                                "entry_time":   zone.entry_time,
                                "park_status":  zone.park_status.value,
                                "linked_zone":  zone.linked_zone,
                                "timestamp":    time.time(),
                            }))
                    except queue.Full:
                        pass

                ocr_submitted.pop(zone_name, None)
                continue

            zone = state_machine.zones.get(zone_name)
            task_entry_time = task.entry_event.get("entry_time")

            if (zone is None
                    or zone.status == ZoneStatus.EMPTY
                    or zone.entry_time != task_entry_time):
                ocr_submitted.pop(zone_name, None)
                continue

            state_machine.set_plate(zone_name, plate)
            ps = zone.plate_status.value

            logger.info(f"[OCR Worker] {zone_name} 완료: {plate} ({ps})")

            entry_event                 = task.entry_event
            entry_event["plate"]        = zone.plate if zone else plate
            entry_event["plate_status"] = ps

            if ps in ("null", "unreadable"):
                entry_event["ocr_error"] = True
                _ensure_image_base64(entry_event, task.snapshot, zone_name, "OCR 실패")
            else:
                entry_event["ocr_error"] = False

            send_queue.put_nowait((PRIORITY_ENTRY, next(_counter), entry_event))
            ocr_submitted.pop(zone_name, None)

            if zone and zone.linked_zone and zone.plate:
                try:
                    send_queue.put_nowait((PRIORITY_UPDATE, next(_counter), {
                        "type":         "plate_update",
                        "zone":         zone.linked_zone,
                        "plate":        zone.plate,
                        "plate_status": "confirmed",
                        "entry_time":   zone.entry_time,
                        "park_status":  zone.park_status.value,
                        "linked_zone":  zone_name,
                        "timestamp":    time.time(),
                    }))
                except queue.Full:
                    pass

        except Exception:
            pass
        finally:
            ocr_queue.task_done()


def send_worker(send_queue: queue.PriorityQueue,
                sender: EventSender,
                stop_event: threading.Event):

    while not stop_event.is_set():
        try:
            priority, _, event = send_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            sender.send(event)
        except Exception:
            pass
        finally:
            send_queue.task_done()


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_BACKUP_PATH), exist_ok=True)

    homography   = HomographyTransformer()
    preprocessor = Preprocessor()
    detector     = VehicleDetector()
    ocr_reader   = PlateReader()
    sender       = EventSender()

    homography.load()

    zone_keys     = list(homography.zones.keys()) if homography.zones else []
    state_machine = ParkingStateMachine(zone_keys)

    _restore_state(state_machine)

    last_mtime     = _get_mtime(ROI_COORDS_PATH)
    CHECK_INTERVAL = 2.0
    last_check     = time.time()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,     1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE,      -3)
    cap.set(cv2.CAP_PROP_BRIGHTNESS,    130)
    cap.set(cv2.CAP_PROP_CONTRAST,      150)
    cap.set(cv2.CAP_PROP_SHARPNESS,     200)

    time.sleep(5.0)
    logger.info("[Camera] C920 자동 포커스 설정 완료")

    if not cap.isOpened():
        sys.exit(1)

    ocr_queue  = queue.PriorityQueue(maxsize=20)
    send_queue = queue.PriorityQueue(maxsize=100)
    stop_event = threading.Event()

    ocr_workers = []
    ocr_submitted: dict[str, bool] = {}
    for i in range(OCR_MAX_THREADS):
        t = threading.Thread(
            target=ocr_worker,
            args=(ocr_queue, send_queue, ocr_reader,
                  state_machine, stop_event, ocr_submitted),
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

    pending_entry: dict[str, dict] = {}

    last_shake_check  = time.time()
    last_snap_cleanup = time.time()
    last_state_backup = time.time()
    last_status_report = time.time()
    last_detect_log    = time.time()

    YOLO_SKIP_FRAMES = 2
    frame_count      = 0
    last_cars        = []
    last_plates      = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            now = time.time()

            if now - last_check >= CHECK_INTERVAL:
                last_check = now
                new_mtime  = _get_mtime(ROI_COORDS_PATH)
                if new_mtime and new_mtime != last_mtime:
                    last_mtime             = new_mtime
                    homography.load()
                    new_keys               = list(homography.zones.keys())
                    state_machine          = ParkingStateMachine(new_keys)
                    pending_entry          = {}

            if (homography.is_ready()
                    and now - last_shake_check >= CAMERA_SHAKE_CHECK_INTERVAL):
                last_shake_check = now
                homography.check_and_auto_correct(frame)

            if now - last_snap_cleanup >= SNAPSHOT_CLEANUP_INTERVAL:
                last_snap_cleanup = now
                _cleanup_snapshots()

            if now - last_state_backup >= STATE_BACKUP_INTERVAL:
                last_state_backup = now
                _backup_state(state_machine)

            if now - last_status_report >= STATUS_REPORT_INTERVAL:
                last_status_report = now
                _report_status(state_machine)

            preprocessor.check_blur(frame)

            enhanced = preprocessor.apply(frame)

            frame_count += 1
            if frame_count % YOLO_SKIP_FRAMES == 0:
                detection_result = detector.detect(enhanced)
                cars             = detection_result["cars"]
                plates           = detection_result["plates"]
                last_cars        = cars
                last_plates      = plates
            else:
                cars   = last_cars
                plates = last_plates

            warped_frame = None
            if homography.is_ready():
                warped_frame = cv2.warpPerspective(
                    frame, homography.matrix,
                    (VIRTUAL_MAP_WIDTH, VIRTUAL_MAP_HEIGHT)
                )

            for zone_name, zone_pts in homography.zones.items():
                zone = state_machine.zones.get(zone_name)
                if zone and zone.empty_snap is None and warped_frame is not None:
                    zone_crop = _get_zone_crop(warped_frame, zone_pts)
                    if zone_crop is not None:
                        state_machine.save_empty_snap(zone_name, zone_crop)

            virtual_cars = []
            if homography.is_ready():
                for car in cars:
                    vx, vy = homography.camera_to_virtual(
                        (car["foot_x"], car["foot_y"])
                    )
                    virtual_cars.append({**car, "vx": vx, "vy": vy})

            confirmed_multi = _check_multi_zone(
                virtual_cars, homography.zones,
                state_machine, send_queue, logger,
                homography_matrix=homography.matrix,
                map_w=VIRTUAL_MAP_WIDTH,
                map_h=VIRTUAL_MAP_HEIGHT,
            )

            for pair in confirmed_multi:
                za  = pair["za"]
                zb  = pair["zb"]
                car = pair["car"]

                for zn in [za, zb]:
                    if ocr_submitted.get(zn, False):
                        continue
                    z = state_machine.zones.get(zn)
                    if z is None:
                        continue

                    multi_entry_event = {
                        "type":         "entry",
                        "zone":         zn,
                        "plate":        None,
                        "plate_status": PlateStatus.PENDING.value,
                        "entry_time":   z.entry_time,
                        "park_status":  "multi_zone",
                        "linked_zone":  zb if zn == za else za,
                        "timestamp":    time.time(),
                        "image_base64": _frame_to_base64(frame),
                    }

                    plate_bbox = detector.find_plate_for_car(car, plates)
                    task = OcrTask(
                        zone_name   = zn,
                        snapshot    = frame.copy(),
                        car_bbox    = car,
                        plate_bbox  = plate_bbox,
                        entry_event = multi_entry_event,
                    )
                    try:
                        ocr_queue.put_nowait((OCR_PRIORITY_ENTRY, next(_counter), task))
                        ocr_submitted[zn] = True
                        pending_entry[zn] = multi_entry_event
                    except queue.Full:
                        pass

            # 구역별 중앙점 감지 결과 (5초마다 1줄 로그용)
            detect_status = {}

            for zone_name, zone_pts in homography.zones.items():
                cars_in_zone = [
                    c for c in virtual_cars
                    if point_in_zone((c["vx"], c["vy"]), zone_pts)
                ]

                zone_obj_for_detect = state_machine.zones.get(zone_name)
                detect_status[zone_name] = (
                    bool(cars_in_zone)
                    or (zone_obj_for_detect is not None
                        and zone_obj_for_detect.status == ZoneStatus.OCCUPIED)
                )

                foot = (cars_in_zone[0]["vx"], cars_in_zone[0]["vy"]) \
                       if cars_in_zone else None

                plate_visible = False
                plate_bbox    = None
                if cars_in_zone:
                    plate_bbox    = detector.find_plate_for_car(cars_in_zone[0], plates)
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
                    if event["type"] == "entry":
                        park_status = event.get("park_status", "normal")

                        image_b64 = None
                        if park_status in ("multi_zone", "aisle_block"):
                            image_b64 = _frame_to_base64(frame)

                        quick_event = {
                            "type":        "entry_quick",
                            "zone":        zone_name,
                            "plate":       None,
                            "park_status": park_status,
                            "linked_zone": event.get("linked_zone"),
                            "entry_time":  event.get("entry_time"),
                            "timestamp":   event["timestamp"],
                        }
                        try:
                            send_queue.put_nowait((PRIORITY_ENTRY, next(_counter), quick_event))
                            logger.info(f"[ENTRY QUICK] {zone_name} PARKED 상태 먼저 전송")
                        except queue.Full:
                            pass

                        event["image_base64"] = image_b64

                        zone_obj = state_machine.zones.get(zone_name)
                        if zone_obj and zone_obj.plate_status in (
                            PlateStatus.UNREADABLE, PlateStatus.NULL
                        ):
                            event["ocr_error"] = True
                        else:
                            event["ocr_error"] = False

                        if not ocr_submitted.get(zone_name, False):
                            task = OcrTask(
                                zone_name   = zone_name,
                                snapshot    = frame.copy(),
                                car_bbox    = cars_in_zone[0] if cars_in_zone else None,
                                plate_bbox  = plate_bbox,
                                entry_event = event,
                            )

                            if not cars_in_zone:
                                event["plate"]        = None
                                event["plate_status"] = PlateStatus.NULL.value
                                event["ocr_error"]    = True
                                _ensure_image_base64(event, frame, zone_name, "차량 없음")
                                try:
                                    send_queue.put_nowait((PRIORITY_ENTRY, next(_counter), event))
                                except queue.Full:
                                    pass
                            else:
                                try:
                                    ocr_queue.put_nowait((OCR_PRIORITY_ENTRY, next(_counter), task))
                                    ocr_submitted[zone_name] = True
                                    pending_entry[zone_name] = event
                                    logger.info(f"[ENTRY] {zone_name} OCR Queue 제출 (대기: {ocr_queue.qsize()})")
                                except queue.Full:
                                    event["plate"]        = None
                                    event["plate_status"] = PlateStatus.NULL.value
                                    event["ocr_error"]    = True
                                    _ensure_image_base64(event, frame, zone_name, "OCR Queue 실패")
                                    try:
                                        send_queue.put_nowait((PRIORITY_ENTRY, next(_counter), event))
                                    except queue.Full:
                                        pass

                    elif event["type"] == "exit":
                        pending = pending_entry.pop(zone_name, None)
                        if pending:
                            pending["plate"]        = None
                            pending["plate_status"] = PlateStatus.NULL.value
                            pending["ocr_error"]    = True
                            _ensure_image_base64(pending, frame, zone_name, "OCR 완료 전 출차")
                            try:
                                send_queue.put_nowait((PRIORITY_ENTRY, next(_counter), pending))
                            except queue.Full:
                                pass

                        ocr_submitted.pop(zone_name, None)

                        try:
                            send_queue.put_nowait((PRIORITY_EXIT, next(_counter), event))
                        except queue.Full:
                            pass

                        logger.info(f"[EXIT] {zone_name} plate={event['plate']}")

                        if event.get("linked_zone"):
                            linked_zn = event["linked_zone"]
                            ocr_submitted.pop(linked_zn, None)
                            pending_entry.pop(linked_zn, None)
                            try:
                                send_queue.put_nowait((PRIORITY_EXIT, next(_counter), {
                                    "type":      "exit",
                                    "zone":      linked_zn,
                                    "exit_time": datetime.fromtimestamp(
                                        event["timestamp"]
                                    ).strftime("%Y-%m-%d %H:%M:%S"),
                                    "timestamp": event["timestamp"],
                                }))
                            except queue.Full:
                                pass

                zone_obj = state_machine.zones.get(zone_name)
                can_recheck = (
                    bool(cars_in_zone) or
                    (zone_obj and zone_obj.status == ZoneStatus.OCCUPIED)
                )
                if state_machine.needs_recheck(zone_name) and can_recheck:

                    cur = state_machine.zones[zone_name]
                    if not ocr_submitted.get(zone_name, False):
                        if not plate_visible:
                            state_machine.mark_rechecked(zone_name)
                        else:
                            recheck_event = {
                                "prev_plate": cur.plate,
                            }
                            task = OcrTask(
                                zone_name   = zone_name,
                                snapshot    = frame.copy(),
                                car_bbox    = cars_in_zone[0],
                                plate_bbox  = plate_bbox,
                                entry_event = recheck_event,
                                is_recheck  = True,
                            )
                            try:
                                ocr_queue.put_nowait((OCR_PRIORITY_RECHECK, next(_counter), task))
                                ocr_submitted[zone_name] = True
                                state_machine.mark_rechecked(zone_name)
                            except queue.Full:
                                state_machine.mark_rechecked(zone_name)

            # 구역별 중앙점 감지 확인 로그 (DETECT_LOG_INTERVAL마다 1줄)
            if now - last_detect_log >= DETECT_LOG_INTERVAL:
                last_detect_log = now
                status_str = " ".join(
                    f"{zn}:{'O' if det else 'X'}"
                    for zn, det in detect_status.items()
                )
                logger.info(f"[DETECT] {status_str}")

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for t in ocr_workers:
            t.join(timeout=5.0)
        send_thread.join(timeout=5.0)

        _backup_state(state_machine)
        cap.release()


def _report_status(state_machine: ParkingStateMachine):
    try:
        full_status = {}
        for name, zone in state_machine.zones.items():
            full_status[name] = {
                "status": zone.status.value,
                "plate":  zone.plate,
            }

        requests.post(
            STATUS_REPORT_URL,
            json={"zones": full_status},
            timeout=10,
        )
    except Exception:
        pass


def _backup_state(state_machine):
    try:
        data = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "zones":    state_machine.to_dict(),
        }
        with open(STATE_BACKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _restore_state(state_machine):
    if not os.path.exists(STATE_BACKUP_PATH):
        return
    try:
        with open(STATE_BACKUP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved_at    = datetime.strptime(data.get("saved_at", ""), "%Y-%m-%d %H:%M:%S")
        age_minutes = (datetime.now() - saved_at).total_seconds() / 60
        if age_minutes > 60:
            return
        zones_data = data.get("zones", {})
        if not zones_data:
            return
        state_machine.from_dict(zones_data)
    except Exception:
        pass


_multi_zone_candidates: dict[str, dict] = {}
_multi_zone_logged: set = set()


def _check_multi_zone(virtual_cars, zones, state_machine,
                      send_queue, logger,
                      homography_matrix=None,
                      map_w=VIRTUAL_MAP_WIDTH,
                      map_h=VIRTUAL_MAP_HEIGHT):
    import cv2 as _cv2
    import numpy as _np

    def get_foot_ends(car, H):
        if H is None:
            return (car["x1"], car["y2"]), (car["x2"], car["y2"])
        pts = _np.array([[
            [float(car["x1"]), float(car["y2"])],
            [float(car["x2"]), float(car["y2"])],
        ]], dtype=_np.float32)
        t = _cv2.perspectiveTransform(pts, H)
        return (float(t[0][0][0]), float(t[0][0][1])), \
               (float(t[0][1][0]), float(t[0][1][1]))

    def deep_in_zone(pt, zone_pts, min_depth=5):
        poly = _np.array(zone_pts, dtype=_np.float32)
        dist = _cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), measureDist=True)
        return dist >= min_depth

    global _multi_zone_candidates, _multi_zone_logged

    confirmed_pairs = []

    zone_main_car = {}
    for zone_name, zone_pts in zones.items():
        for car in virtual_cars:
            if point_in_zone((car["vx"], car["vy"]), zone_pts):
                zone_main_car[zone_name] = car
                break

    now = time.time()
    detected_pairs = set()

    for zone_name, zone_pts in zones.items():
        if zone_name in zone_main_car:
            continue
        for car in virtual_cars:
            foot_left, foot_right = get_foot_ends(car, homography_matrix)
            if (deep_in_zone(foot_left,  zone_pts, min_depth=5) or
                    deep_in_zone(foot_right, zone_pts, min_depth=5)):

                main_zone = None
                for mz, mc in zone_main_car.items():
                    if mc is car:
                        main_zone = mz
                        break
                if main_zone is None:
                    continue

                pair_key = f"{main_zone}+{zone_name}"

                if pair_key not in _multi_zone_candidates:
                    _multi_zone_candidates[pair_key] = {
                        "since":      now,
                        "main_zone":  main_zone,
                        "extra_zone": zone_name,
                        "car":        car,
                        "hit_frames": 0,
                        "last_seen":  now,
                    }
                    _multi_zone_logged.add(pair_key)
                else:
                    _multi_zone_candidates[pair_key]["hit_frames"] += 1
                    _multi_zone_candidates[pair_key]["last_seen"]   = now
                    _multi_zone_candidates[pair_key]["car"]         = car

                detected_pairs.add(pair_key)

                hit_frames = _multi_zone_candidates[pair_key].get("hit_frames", 0)
                if hit_frames >= 3:
                    za = main_zone
                    zb = zone_name

                    state_a = state_machine.zones.get(za)
                    state_b = state_machine.zones.get(zb)

                    if state_a is None or state_b is None:
                        continue

                    if (state_a.status == ZoneStatus.OCCUPIED and
                            state_b.status == ZoneStatus.OCCUPIED and
                            state_a.linked_zone == zb and
                            state_b.linked_zone == za):
                        continue

                    if (state_a.status == ZoneStatus.OCCUPIED and
                            state_b.status == ZoneStatus.OCCUPIED):
                        continue

                    state_machine.set_multi_zone(za, zb, None)

                    za_zone        = state_machine.zones.get(za)
                    entry_time_str = datetime.fromtimestamp(
                        za_zone.entry_time
                    ).strftime("%Y-%m-%d %H:%M:%S")

                    for zn in [za, zb]:
                        other_zone = zb if zn == za else za
                        try:
                            send_queue.put_nowait((PRIORITY_ENTRY, next(_counter), {
                                "type":        "entry_quick",
                                "zone":        zn,
                                "plate":       None,
                                "park_status": "multi_zone",
                                "linked_zone": other_zone,
                                "entry_time":  entry_time_str,
                                "timestamp":   time.time(),
                            }))
                        except queue.Full:
                            pass

                    confirmed_pairs.append({
                        "za":  za,
                        "zb":  zb,
                        "car": _multi_zone_candidates[pair_key]["car"],
                    })

                    _multi_zone_candidates.pop(pair_key, None)
                    _multi_zone_logged.discard(pair_key)

    for key in list(_multi_zone_candidates.keys()):
        if key not in detected_pairs:
            last_seen = _multi_zone_candidates[key].get("last_seen", 0)
            if now - last_seen > 3.0:
                _multi_zone_candidates.pop(key, None)
                _multi_zone_logged.discard(key)

    return confirmed_pairs


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
                if datetime.fromtimestamp(os.path.getmtime(fpath)) < cutoff:
                    os.remove(fpath)
                    deleted += 1
            except Exception:
                continue
    except Exception:
        pass
    return deleted


def _get_mtime(path):
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    main()