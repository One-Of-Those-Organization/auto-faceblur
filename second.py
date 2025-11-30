#!/usr/bin/env python3
import os
import cv2
import numpy as np
from deepface import DeepFace
import math
import time
from ultralytics import YOLO
import tensorflow as tf
import torch

# ============================================================
# CONFIGURATION
# ============================================================
SKIP_FRAMES = 30
THRESHOLD = 0.40
INPUT_RES = (640, 480)
USE_GPU = False  # Set False untuk CPU

# ============================================================
# SETUP HARDWARE
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
cuda_fix_path = os.path.join(base_dir, "cuda_fix")
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_fix_path}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("INFO: Mode CPU Selected.")
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# ============================================================
# LOAD MODEL
# ============================================================
print("INFO: Loading YOLO...")
model = YOLO("model.pt")

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
# ============================================================
print("INFO: Memuat whitelist...")
whitelist_path = "whitelist/"
target_embeddings = []

if not os.path.exists(whitelist_path):
    os.makedirs(whitelist_path)

files = [os.path.join(whitelist_path, f) for f in os.listdir(whitelist_path)
         if os.path.isfile(os.path.join(whitelist_path, f))]

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
# ============================================================
def get_distance(emb1, emb2):
    a = np.matmul(emb1, emb2)
    b = np.linalg.norm(emb1)
    c = np.linalg.norm(emb2)
    return 1 - (a / (b * c))

def blur_face(face):
    return cv2.GaussianBlur(face, (51, 51), 30)

def is_close(box1, box2, limit=50):
    cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    cx2, cy2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < limit

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
frame_count = 0
tracked_faces = []

fps = 0
prev_time = time.time()

print("INFO: Kamera berjalan. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sframe = cv2.resize(frame, INPUT_RES)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(sframe, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Deteksi YOLO menggunakan target device (cpu/0)
    results = model(sframe, stream=True, verbose=False, device=DEVICE_TARGET)
    current_faces_status = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w, _ = sframe.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = sframe[y1:y2, x1:x2]
            if face_img.size == 0: continue

            is_whitelisted = False
            needs_recognition = (frame_count % SKIP_FRAMES == 0)
            matched_prev_status = None
            debug_dist = 0.0

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

                        if min_dist <= THRESHOLD:
                            is_whitelisted = True
                except Exception:
                    pass

            current_faces_status.append({
                "box": [x1, y1, x2, y2], 
                "status": is_whitelisted,
                "dist": debug_dist 
            })

            if not is_whitelisted:
                sframe[y1:y2, x1:x2] = blur_face(face_img)
                color = (0, 0, 255)
            else:
                cv2.rectangle(sframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
                color = (0, 255, 0)

            cv2.putText(sframe, f"Dist: {debug_dist:.3f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    tracked_faces = current_faces_status
    frame_count += 1

    cv2.imshow("YOLOv11 Face Blur DEBUG", sframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()