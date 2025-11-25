#!/usr/bin/env python3
import cv2
import numpy as np
from deepface import DeepFace
import os
import urllib.request
import math

# --- 1. CONFIGURATION ---
SKIP_FRAMES = 30
THRESHOLD = 0.55
INPUT_RES = (640, 480)   # Resolusi kamera (Standard VGA)

# --- 2. SETUP MODEL YUNET ---
yunet_weights = "face_detection_yunet_2023mar.onnx"
yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

if not os.path.exists(yunet_weights):
    print("INFO: Mengunduh model YuNet...")
    try:
        urllib.request.urlretrieve(yunet_url, yunet_weights)
    except Exception as e:
        print(f"ERROR: Download gagal: {e}")
        exit()

face_detector = cv2.FaceDetectorYN.create(
        model=yunet_weights,
        config="",
        input_size=INPUT_RES,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
        )

# --- 3. LOAD WHITELIST ---
print("INFO: Memuat whitelist...")
directory_path = 'whitelist/'
target_embeddings = []

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

for img_path in files:
    try:
        # Gunakan VGG-Face
        embedding_result = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                enforce_detection=False
                )
        if embedding_result:
            target_embeddings.append(embedding_result[0]["embedding"])
            print(f"  + Loaded: {os.path.basename(img_path)}")
    except Exception as e:
        print(f"  - Gagal load {img_path}: {e}")

print(f"INFO: Total {len(target_embeddings)} wajah di whitelist.")
target_embeddings = np.array(target_embeddings) # Convert ke numpy array untuk performa

# --- 4. HELPER FUNCTIONS ---
def get_distance(emb1, emb2):
    """Hitung Cosine Distance antara dua embedding"""
    a = np.matmul(emb1, emb2)
    b = np.linalg.norm(emb1)
    c = np.linalg.norm(emb2)
    return 1 - (a / (b * c))

def is_close(box1, box2, limit=50):
    """Cek apakah posisi wajah sekarang dekat dengan posisi wajah yang sudah dikenali sebelumnya"""
    # Hitung titik tengah (centroid)
    cx1 = box1[0] + (box1[2] // 2)
    cy1 = box1[1] + (box1[3] // 2)
    cx2 = box2[0] + (box2[2] // 2)
    cy2 = box2[1] + (box2[3] // 2)

    dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return dist < limit

# --- 5. MAIN LOOP ---
cap = cv2.VideoCapture(0)
frame_count = 0

# Variabel untuk menyimpan status wajah agar tidak perlu cek tiap frame
# Format: [{'box': [x,y,w,h], 'status': True/False}]
tracked_faces = []

print("INFO: Mulai kamera. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break

    if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Resize frame
    sframe = cv2.resize(frame, INPUT_RES, interpolation=cv2.INTER_LINEAR)
    h_img, w_img, _ = sframe.shape
    face_detector.setInputSize((w_img, h_img))

    # 1. Deteksi Wajah (Cepat - Jalan tiap frame)
    _, faces = face_detector.detect(sframe)

    current_faces_status = [] # List untuk menyimpan status wajah di frame INI

    if faces is not None:
        for face in faces:
            box = face[0:4].astype(np.int32)
            x, y, w, h = max(0, box[0]), max(0, box[1]), min(box[2], w_img-box[0]), min(box[3], h_img-box[1])

            if w <= 0 or h <= 0: continue

            face_img = sframe[y:y+h, x:x+w]
            is_whitelisted = False

            # --- LOGIKA OPTIMASI ---
            # Kita hanya menjalankan DeepFace (BERAT) jika:
            # A. Ini adalah frame ke-30 (Scanning ulang berkala)
            # B. Wajah ini belum ada di data tracking (Wajah baru muncul)

            needs_recognition = (frame_count % SKIP_FRAMES == 0)

            # Cek apakah wajah ini sudah ada di tracking list sebelumnya?
            matched_prev_status = None
            if not needs_recognition:
                for tf in tracked_faces:
                    if is_close(box, tf['box']):
                        matched_prev_status = tf['status']
                        break

                if matched_prev_status is not None:
                    is_whitelisted = matched_prev_status
                else:
                    # Jika wajah baru muncul tapi bukan jadwal scan, paksa scan
                    needs_recognition = True

            # --- DEEPFACE RECOGNITION (BERAT) ---
            if needs_recognition and len(target_embeddings) > 0:
                try:
                    # Visual feedback text "Scanning..."
                    cv2.putText(sframe, ".", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    current_res = DeepFace.represent(img_path=face_img, model_name="Facenet512", enforce_detection=False)
                    if current_res:
                        curr_emb = current_res[0]["embedding"]

                        # Cek jarak dengan semua whitelist
                        min_dist = 100
                        for target in target_embeddings:
                            dist = get_distance(target, curr_emb)
                            if dist < min_dist: min_dist = dist

                        print(f"DEBUG: Jarak wajah = {min_dist:.4f} (Threshold: {THRESHOLD})")

                        if min_dist <= THRESHOLD:
                            is_whitelisted = True
                except Exception as e:
                    pass

            # Simpan status untuk frame berikutnya
            current_faces_status.append({'box': box, 'status': is_whitelisted})

            # --- VISUALISASI ---
            if not is_whitelisted:
                blur_roi = cv2.GaussianBlur(face_img, (51, 51), 30)
                sframe[y:y+h, x:x+w] = blur_roi

            """
            if is_whitelisted:
                # UNLOCK (Hijau)
                cv2.rectangle(sframe, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(sframe, "OPEN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # LOCKED (Blur)
                try:
                    # Blur yang sangat kuat
                    blur_roi = cv2.GaussianBlur(face_img, (51, 51), 30)
                    sframe[y:y+h, x:x+w] = blur_roi
                except: pass
            """

    # Update tracking list
    tracked_faces = current_faces_status

    frame_count += 1
    cv2.imshow('Smart Face Blur', sframe)

cap.release()
cv2.destroyAllWindows()
