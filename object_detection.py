"""
Object Detection menggunakan YOLOv8
Mendeteksi: person, bicycle, car, motorcycle, bus, truck
Fitur tambahan:
  - Deteksi kendaraan DIAM (stationary)
  - Deteksi kendaraan LAWAN ARAH (wrong-way) berdasarkan jalur
Video input: 2026-02-25 235836.mov
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ============================================================
# Konfigurasi
# ============================================================
VIDEO_PATH = "2026-02-25 235836.mov"
OUTPUT_PATH = "output_detection.avi"
MODEL_NAME = "yolov8m.pt"  

# Class ID yang akan dideteksi (sesuai COCO dataset)
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Kelas kendaraan (untuk deteksi diam & lawan arah, tanpa person)
VEHICLE_CLASSES = {1, 2, 3, 5, 7}

# Warna bounding box untuk setiap kelas (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),      # person     - hijau
    1: (255, 165, 0),    # bicycle    - oranye
    2: (255, 0, 0),      # car        - biru
    3: (0, 0, 255),      # motorcycle - merah
    5: (255, 255, 0),    # bus        - cyan
    7: (128, 0, 128),    # truck      - ungu
}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# ============================================================
# Konfigurasi Tracking & Deteksi Perilaku
# ============================================================
MAX_MATCH_DISTANCE = 100
TRACK_HISTORY_LENGTH = 30

# --- Deteksi Kendaraan Diam ---
STATIONARY_DISPLACEMENT_THRESHOLD = 15  # piksel
STATIONARY_FRAME_COUNT = 20

# ============================================================
# Konfigurasi Jalur Jalan (Lane)
# ============================================================
# Garis pembatas median didefinisikan oleh 2 titik (perspektif):
#   - Titik atas (dekat vanishing point): MEDIAN_TOP = (x, y)
#   - Titik bawah (dekat kamera):         MEDIAN_BOTTOM = (x, y)
# Kendaraan di KIRI median normalnya bergerak MENJAUH dari kamera (ke atas frame)
# Kendaraan di KANAN median normalnya bergerak MENDEKATI kamera (ke bawah frame)

MEDIAN_TOP = (1000, 200)
MEDIAN_BOTTOM = (1050, 1080)

# Arah normal per jalur (dalam derajat, 0=kanan, 90=bawah, 180=kiri, 270=atas)
# Jalur kiri: kendaraan bergerak menjauh (ke atas frame, ~270°)
# Jalur kanan: kendaraan bergerak mendekat (ke bawah frame, ~90°)
LEFT_LANE_NORMAL_DIRECTION = 270   # menjauh dari kamera (atas)
RIGHT_LANE_NORMAL_DIRECTION = 90   # mendekati kamera (bawah)

# Toleransi sudut untuk lawan arah (derajat)
WRONG_WAY_ANGLE_TOLERANCE = 60

# Minimum perpindahan (piksel) untuk bisa menentukan arah gerak
MIN_DISPLACEMENT_FOR_DIRECTION = 25


# ============================================================
# Fungsi Lane
# ============================================================
def get_median_x_at_y(y):
    """Hitung posisi X dari garis median pada koordinat Y tertentu (interpolasi linear)."""
    x_top, y_top = MEDIAN_TOP
    x_bot, y_bot = MEDIAN_BOTTOM
    if y_bot == y_top:
        return x_top
    t = (y - y_top) / (y_bot - y_top)
    return x_top + t * (x_bot - x_top)


def get_lane(cx, cy):
    """Tentukan jalur kendaraan: 'left' atau 'right' berdasarkan posisi relatif terhadap median."""
    median_x = get_median_x_at_y(cy)
    return "left" if cx < median_x else "right"


def get_normal_direction_for_lane(lane):
    """Ambil arah normal lalu lintas untuk jalur tertentu."""
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
    """Cek apakah kendaraan diam."""
    if len(track_history) < STATIONARY_FRAME_COUNT:
        return False
    recent = list(track_history)[-STATIONARY_FRAME_COUNT:]
    first = np.array(recent[0])
    max_disp = max(np.linalg.norm(np.array(p) - first) for p in recent)
    return max_disp < STATIONARY_DISPLACEMENT_THRESHOLD


def get_movement_angle(track_history):
    """Hitung sudut arah gerak kendaraan (dalam derajat)."""
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
    """
    Cek apakah kendaraan bergerak lawan arah berdasarkan jalurnya.
    1. Tentukan kendaraan ada di jalur kiri atau kanan (relatif terhadap median)
    2. Ambil arah normal untuk jalur tersebut
    3. Bandingkan arah gerak kendaraan dengan arah normal
    """
    angle, displacement = get_movement_angle(track_history)
    if angle is None:
        return False

    # Tentukan jalur berdasarkan posisi kendaraan
    lane = get_lane(cx, cy)
    normal_dir = get_normal_direction_for_lane(lane)

    # Hitung perbedaan sudut
    diff = abs(angle - normal_dir)
    if diff > 180:
        diff = 360 - diff

    return diff > (180 - WRONG_WAY_ANGLE_TOLERANCE)


def draw_status_label(frame, text, x1, y1, x2, color, y_offset=0):
    """Gambar label status di atas bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y = y1 - 25 - y_offset
    cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 4, label_y + 3), color, -1)
    cv2.putText(frame, text, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_median_line(frame):
    """Gambar garis median pada frame untuk visualisasi."""
    cv2.line(frame, MEDIAN_TOP, MEDIAN_BOTTOM, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "MEDIAN", (MEDIAN_TOP[0] + 10, MEDIAN_TOP[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


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

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    tracker = CentroidTracker()
    frame_count = 0

    print("[INFO] Memulai object detection... (Tekan 'q' untuk keluar)")

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

        # Gambar garis median
        draw_median_line(frame)

        frame_stationary = 0
        frame_wrong_way = 0

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
                        frame_stationary += 1
                    if vehicle_wrong_way:
                        frame_wrong_way += 1

            # Pilih warna box
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

            # Label dengan jalur info
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
            f"Kendaraan Diam: {frame_stationary}",
            f"Lawan Arah: {frame_wrong_way}",
        ]
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 30
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (5, y_pos - th - 5), (15 + tw, y_pos + 5), (0, 0, 0), -1)
            if i == 1 and frame_stationary > 0:
                text_color = (0, 165, 255)
            elif i == 2 and frame_wrong_way > 0:
                text_color = (0, 0, 255)
            else:
                text_color = (0, 255, 255)
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        out.write(frame)

        display_width = 1280
        scale = display_width / frame_width
        display_height = int(frame_height * scale)
        display_frame = cv2.resize(frame, (display_width, display_height))

        cv2.imshow("YOLO Object Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Dihentikan oleh user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Video output tersimpan di: {OUTPUT_PATH}")
    print("[INFO] Selesai!")


if __name__ == "__main__":
    main()
