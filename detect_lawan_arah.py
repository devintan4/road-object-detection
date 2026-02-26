"""
Clip Extractor: Hanya menampilkan & menyimpan bagian video
yang mengandung deteksi kendaraan LAWAN ARAH (berdasarkan jalur).

Menggunakan logika lane-aware yang sama dengan object_detection.py
tetapi hanya menulis frame yang ada deteksi lawan arah.
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ============================================================
# Konfigurasi
# ============================================================
VIDEO_PATH = "2026-02-25 235836.mov"
OUTPUT_PATH = "output_lawan_arah.avi"
MODEL_NAME = "yolov8m.pt"

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASSES = {1, 2, 3, 5, 7}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 165, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    5: (255, 255, 0),
    7: (128, 0, 128),
}

CONFIDENCE_THRESHOLD = 0.5

# ============================================================
# Konfigurasi Tracking
# ============================================================
MAX_MATCH_DISTANCE = 100
TRACK_HISTORY_LENGTH = 30

STATIONARY_DISPLACEMENT_THRESHOLD = 15
STATIONARY_FRAME_COUNT = 20

# ============================================================
# Konfigurasi Jalur Jalan (Lane)
# ============================================================
# Garis median (perspektif): dari titik atas ke titik bawah frame
MEDIAN_TOP = (1000, 200)
MEDIAN_BOTTOM = (1050, 1080)

# Arah normal per jalur
LEFT_LANE_NORMAL_DIRECTION = 270   # menjauh dari kamera (atas)
RIGHT_LANE_NORMAL_DIRECTION = 90   # mendekati kamera (bawah)

WRONG_WAY_ANGLE_TOLERANCE = 60
MIN_DISPLACEMENT_FOR_DIRECTION = 25

# Buffer frame sebelum & sesudah event lawan arah
BUFFER_FRAMES_BEFORE = 30
BUFFER_FRAMES_AFTER = 60


# ============================================================
# Fungsi Lane
# ============================================================
def get_median_x_at_y(y):
    x_top, y_top = MEDIAN_TOP
    x_bot, y_bot = MEDIAN_BOTTOM
    if y_bot == y_top:
        return x_top
    t = (y - y_top) / (y_bot - y_top)
    return x_top + t * (x_bot - x_top)


def get_lane(cx, cy):
    median_x = get_median_x_at_y(cy)
    return "left" if cx < median_x else "right"


def get_normal_direction_for_lane(lane):
    if lane == "left":
        return LEFT_LANE_NORMAL_DIRECTION
    else:
        return RIGHT_LANE_NORMAL_DIRECTION


# ============================================================
# Simple Centroid Tracker
# ============================================================
class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
        self.track_classes = {}
        self.disappeared = {}
        self.max_disappeared = 15

    def update(self, detections):
        if len(detections) == 0:
            to_delete = []
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
                del self.track_classes[tid]
                del self.disappeared[tid]
            return {}

        det_centroids = np.array([(d[0], d[1]) for d in detections])

        if len(self.tracks) == 0:
            result = {}
            for det in detections:
                cx, cy, cls_id, x1, y1, x2, y2, conf = det
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = deque(maxlen=TRACK_HISTORY_LENGTH)
                self.tracks[tid].append((cx, cy))
                self.track_classes[tid] = cls_id
                self.disappeared[tid] = 0
                result[tid] = det
            return result

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid][-1] for tid in track_ids])

        dist_matrix = np.linalg.norm(
            track_centroids[:, np.newaxis] - det_centroids[np.newaxis, :], axis=2
        )

        matched_tracks = set()
        matched_dets = set()
        result = {}

        pairs = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                pairs.append((dist_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0])

        for dist, i, j in pairs:
            if i in matched_tracks or j in matched_dets:
                continue
            if dist > MAX_MATCH_DISTANCE:
                break
            tid = track_ids[i]
            det = detections[j]
            self.tracks[tid].append((det[0], det[1]))
            self.track_classes[tid] = det[2]
            self.disappeared[tid] = 0
            matched_tracks.add(i)
            matched_dets.add(j)
            result[tid] = det

        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.track_classes[tid]
                    del self.disappeared[tid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = deque(maxlen=TRACK_HISTORY_LENGTH)
                self.tracks[tid].append((det[0], det[1]))
                self.track_classes[tid] = det[2]
                self.disappeared[tid] = 0
                result[tid] = det

        return result


def is_stationary(track_history):
    if len(track_history) < STATIONARY_FRAME_COUNT:
        return False
    recent = list(track_history)[-STATIONARY_FRAME_COUNT:]
    first = np.array(recent[0])
    max_disp = max(np.linalg.norm(np.array(p) - first) for p in recent)
    return max_disp < STATIONARY_DISPLACEMENT_THRESHOLD


def get_movement_angle(track_history):
    if len(track_history) < 5:
        return None, 0
    recent = list(track_history)[-min(15, len(track_history)):]
    first = np.array(recent[0])
    last = np.array(recent[-1])
    displacement = np.linalg.norm(last - first)
    if displacement < MIN_DISPLACEMENT_FOR_DIRECTION:
        return None, displacement
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    return angle, displacement


def is_wrong_way(track_history, cx, cy):
    """Cek lawan arah berdasarkan jalur kendaraan."""
    angle, displacement = get_movement_angle(track_history)
    if angle is None:
        return False
    lane = get_lane(cx, cy)
    normal_dir = get_normal_direction_for_lane(lane)
    diff = abs(angle - normal_dir)
    if diff > 180:
        diff = 360 - diff
    return diff > (180 - WRONG_WAY_ANGLE_TOLERANCE)


def draw_status_label(frame, text, x1, y1, x2, color, y_offset=0):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y = y1 - 25 - y_offset
    cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 4, label_y + 3), color, -1)
    cv2.putText(frame, text, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_median_line(frame):
    cv2.line(frame, MEDIAN_TOP, MEDIAN_BOTTOM, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "MEDIAN", (MEDIAN_TOP[0] + 10, MEDIAN_TOP[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def draw_detections(frame, tracked_objects, tracker, frame_count, total_frames):
    """Gambar semua deteksi dan return (wrong_way_count, stationary_count)."""
    wrong_way_count = 0
    stationary_count = 0

    draw_median_line(frame)

    for tid, det in tracked_objects.items():
        cx, cy, cls_id, x1, y1, x2, y2, confidence = det
        class_name = TARGET_CLASSES[cls_id]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))

        vehicle_stationary = False
        vehicle_wrong_way = False

        if cls_id in VEHICLE_CLASSES:
            track_hist = tracker.tracks.get(tid)
            if track_hist:
                vehicle_stationary = is_stationary(track_hist)
                vehicle_wrong_way = is_wrong_way(track_hist, cx, cy)
                if vehicle_stationary:
                    stationary_count += 1
                if vehicle_wrong_way:
                    wrong_way_count += 1

        if vehicle_wrong_way:
            box_color = (0, 0, 255)
            thickness = 3
        elif vehicle_stationary:
            box_color = (0, 165, 255)
            thickness = 3
        else:
            box_color = color
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        lane = get_lane(cx, cy)
        lane_label = "L" if lane == "left" else "R"
        label = f"ID{tid} {class_name}: {confidence:.2f} [{lane_label}]"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 5), (x1 + tw, y1), box_color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_off = 0
        if vehicle_wrong_way:
            draw_status_label(frame, "LAWAN ARAH!", x1, y1, x2, (0, 0, 255), y_off)
            y_off += 25
        if vehicle_stationary:
            draw_status_label(frame, "DIAM", x1, y1, x2, (0, 165, 255), y_off)

    # Info panel
    info_lines = [
        f"Frame: {frame_count}/{total_frames}",
        f"Kendaraan Diam: {stationary_count}",
        f"Lawan Arah: {wrong_way_count}",
    ]
    for i, line in enumerate(info_lines):
        y_pos = 30 + i * 30
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (5, y_pos - th - 5), (15 + tw, y_pos + 5), (0, 0, 0), -1)
        if i == 1 and stationary_count > 0:
            text_color = (0, 165, 255)
        elif i == 2 and wrong_way_count > 0:
            text_color = (0, 0, 255)
        else:
            text_color = (0, 255, 255)
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    return wrong_way_count, stationary_count


def main():
    print(f"[INFO] Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print(f"[INFO] Membuka video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka video: {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Resolusi: {frame_width}x{frame_height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total frame: {total_frames}")
    print(f"[INFO] Jalur kiri  -> arah normal: {LEFT_LANE_NORMAL_DIRECTION}° (menjauh)")
    print(f"[INFO] Jalur kanan -> arah normal: {RIGHT_LANE_NORMAL_DIRECTION}° (mendekat)")
    print(f"[INFO] Output hanya berisi segmen dengan kendaraan LAWAN ARAH")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    tracker = CentroidTracker()

    frame_buffer = deque(maxlen=BUFFER_FRAMES_BEFORE)
    after_counter = 0
    buffer_flushed = False

    frame_count = 0
    total_wrong_way_events = 0
    total_written_frames = 0
    segment_count = 0

    print("[INFO] Memulai scanning video untuk deteksi lawan arah... (Tekan 'q' untuk keluar)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video selesai.")
            break

        frame_count += 1

        results = model(frame, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, cls_id, x1, y1, x2, y2, confidence))

        tracked_objects = tracker.update(detections)

        # Gambar deteksi
        annotated = frame.copy()
        frame_wrong_way, frame_stationary = draw_detections(
            annotated, tracked_objects, tracker, frame_count, total_frames
        )

        # Label segmen
        if frame_wrong_way > 0 or after_counter > 0:
            cv2.putText(
                annotated, ">> SEGMEN LAWAN ARAH <<", (frame_width // 2 - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3
            )

        # Logika penulisan segmen
        if frame_wrong_way > 0:
            if not buffer_flushed:
                segment_count += 1
                print(f"[EVENT] Segmen #{segment_count} - Lawan arah terdeteksi di frame {frame_count}")
                for buffered_frame in frame_buffer:
                    out.write(buffered_frame)
                    total_written_frames += 1
                buffer_flushed = True
                frame_buffer.clear()

            out.write(annotated)
            total_written_frames += 1
            after_counter = BUFFER_FRAMES_AFTER
            total_wrong_way_events += 1

        elif after_counter > 0:
            out.write(annotated)
            total_written_frames += 1
            after_counter -= 1
            if after_counter == 0:
                buffer_flushed = False
                separator = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(
                    separator, f"-- Akhir Segmen #{segment_count} --",
                    (frame_width // 2 - 200, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
                )
                for _ in range(10):
                    out.write(separator)
                    total_written_frames += 1
        else:
            frame_buffer.append(annotated)

        if frame_count % 500 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"[PROGRESS] {frame_count}/{total_frames} ({pct:.1f}%) | "
                  f"Segmen: {segment_count} | Frame output: {total_written_frames}")

        display_width = 1280
        scale = display_width / frame_width
        display_height = int(frame_height * scale)
        display_frame = cv2.resize(annotated, (display_width, display_height))
        cv2.imshow("YOLO - Wrong Way Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Dihentikan oleh user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"[HASIL]")
    print(f"  Total frame diproses  : {frame_count}")
    print(f"  Segmen lawan arah     : {segment_count}")
    print(f"  Frame events          : {total_wrong_way_events}")
    print(f"  Frame output          : {total_written_frames}")
    print(f"  Video output          : {OUTPUT_PATH}")
    print(f"{'='*50}")

    if segment_count == 0:
        print("\n[INFO] Tidak ada kendaraan lawan arah terdeteksi.")
        print("[TIP]  Coba sesuaikan MEDIAN_TOP/MEDIAN_BOTTOM atau turunkan threshold.")
    else:
        print(f"\n[INFO] Video tersimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
