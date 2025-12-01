#!/usr/bin/env python3
# ============================================================
# AUTO FACE BLUR SYSTEM USING YOLO + DEEPFACE
# Dokumentasi tambahan: menjelaskan fungsi setiap bagian kode
# ============================================================

import os
import cv2
import numpy as np
from deepface import DeepFace
import math
import time
from ultralytics import YOLO
import tensorflow as tf

# ============================================================
# CONFIGURATION SECTION
# Bagian konfigurasi untuk parameter utama sistem
# ============================================================
SKIP_FRAMES = 30           # Interval frame untuk pengenalan wajah (DeepFace)
THRESHOLD = 0.40           # Batas kemiripan embedding wajah
INPUT_RES = (640, 480)     # Resolusi input kamera
USE_GPU = False            # Opsi penggunaan GPU untuk YOLO/DeepFace

# ============================================================
# HARDWARE & CUDA SETUP
# Bagian pengaturan environment TensorFlow dan CUDA
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
cuda_fix_path = os.path.join(base_dir, "cuda_fix")
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_fix_path}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Sembunyikan warning TF

# Nonaktifkan GPU jika USE_GPU = False
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("INFO: Mode CPU Selected.")
else:
    # Setup memory growth jika GPU aktif
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# ============================================================
# LOAD YOLO MODEL
# Bagian untuk memuat model deteksi wajah YOLO
# ============================================================
print("INFO: Loading YOLO...")
model = YOLO("model.pt")      # Model YOLO custom

# Pindahkan model ke CPU/GPU
if USE_GPU:
    model.to('cuda')
    DEVICE_TARGET = 0
    print("INFO: YOLO running on GPU")
else:
    model.to('cpu')
    DEVICE_TARGET = 'cpu'
    print("INFO: YOLO running on CPU")

# ============================================================
# LOAD WHITELIST

# Bagian untuk memuat embedding wajah yang diizinkan (tidak diblur)
# ============================================================
print("INFO: Memuat whitelist...")
whitelist_path = "whitelist/"
target_embeddings = []

# Buat folder whitelist jika belum ada
if not os.path.exists(whitelist_path):
    os.makedirs(whitelist_path)

# Ambil semua file di folder whitelist
files = [os.path.join(whitelist_path, f) for f in os.listdir(whitelist_path)
         if os.path.isfile(os.path.join(whitelist_path, f))]

# Generate embedding tiap foto whitelist
for img_path in files:
    try:
        emb = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet512",
            enforce_detection=True
        )
        if emb:
            target_embeddings.append(emb[0]["embedding"])
            print(f"  + Loaded: {os.path.basename(img_path)}")
    except Exception as e:
        print(f"  - Gagal load {img_path}: {e}")

target_embeddings = np.array(target_embeddings)

# ============================================================
# HELPER FUNCTIONS
# Berisi fungsi kecil pendukung untuk perhitungan & blur wajah
# ============================================================

def get_distance(emb1, emb2):
    """Hitung cosine distance antara dua embedding wajah"""
    a = np.matmul(emb1, emb2)
    b = np.linalg.norm(emb1)
    c = np.linalg.norm(emb2)
    return 1 - (a / (b * c))


def blur_face(face):
    """Fungsi untuk melakukan blur wajah dengan Gaussian Blur"""
    return cv2.GaussianBlur(face, (51, 51), 30)


def is_close(box1, box2, limit=50):
    """Cek apakah dua bounding box berdekatan (tracking sederhana)"""
    cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    cx2, cy2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < limit

# ============================================================
# MAIN LOOP (PROGRAM UTAMA)
# Meng-handle kamera, deteksi YOLO, pengenalan wajah, dan blur
# ============================================================
cap = cv2.VideoCapture(0)      # Akses kamera default
frame_count = 0                # Counter frame untuk SKIP_FRAMES
tracked_faces = []             # Penyimpanan status wajah frame sebelumnya

fps = 0
prev_time = time.time()

print("INFO: Kamera berjalan. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()   # Ambil frame dari kamera
    if not ret:
        break

    sframe = cv2.resize(frame, INPUT_RES)   # Resize ke resolusi input

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(sframe, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Deteksi wajah menggunakan YOLO
    results = model(sframe, stream=True, verbose=False, device=DEVICE_TARGET)
    current_faces_status = []

    for result in results:
        for box in result.boxes:
            # Ambil bounding box wajah
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Pastikan bounding box tidak keluar frame
            h, w, _ = sframe.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = sframe[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Variabel status
            is_whitelisted = False
            needs_recognition = (frame_count % SKIP_FRAMES == 0)  # Evaluasi ulang setiap SKIP_FRAMES
            matched_prev_status = None
            debug_dist = 0.0

            # Tracking sederhana: apakah wajah ini sama dengan sebelumnya?
            if not needs_recognition:
                for tf_data in tracked_faces:
                    if is_close([x1, y1, x2, y2], tf_data["box"]):
                        matched_prev_status = tf_data["status"]
                        debug_dist = tf_data.get("dist", 0.0)
                        break

                if matched_prev_status is not None:
                    is_whitelisted = matched_prev_status
                else:
                    needs_recognition = True

            # Pengenalan wajah menggunakan DeepFace jika diperlukan
            if needs_recognition and len(target_embeddings) > 0:
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                try:
                    curr = DeepFace.represent(
                        img_path=face_img_rgb,
                        model_name="Facenet512",
                        enforce_detection=False
                    )
                    if curr:
                        curr_emb = curr[0]["embedding"]
                        dists = [get_distance(t, curr_emb) for t in target_embeddings]
                        min_dist = min(dists)
                        debug_dist = min_dist

                        print(f"[Check] Dist: {min_dist:.4f} | Thresh: {THRESHOLD}")

                        # Jika embedding cukup mirip → whitelist
                        if min_dist <= THRESHOLD:
                            is_whitelisted = True
                except Exception:
                    pass

            # Simpan status wajah untuk frame berikutnya
            current_faces_status.append({
                "box": [x1, y1, x2, y2],
                "status": is_whitelisted,
                "dist": debug_dist
            })

            # Blur atau tampilkan bounding box
            if not is_whitelisted:
                sframe[y1:y2, x1:x2] = blur_face(face_img)  # Blur wajah
                color = (0, 0, 255)
            else:
                cv2.rectangle(sframe, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Wajah dikenal
                color = (0, 255, 0)

            # Tampilkan nilai distance untuk debugging
            cv2.putText(sframe, f"Dist: {debug_dist:.3f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Update tracked faces
    tracked_faces = current_faces_status
    frame_count += 1

    # Tampilkan frame ke layar
    cv2.imshow("YOLOv11 Face Blur DEBUG", sframe)

    # Tombol keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# CLEANUP SECTION
# Tutup kamera dan jendela OpenCV saat program selesai
# ============================================================
cap.release()
cv2.destroyAllWindows()