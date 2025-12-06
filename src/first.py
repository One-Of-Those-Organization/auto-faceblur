#!/usr/bin/env python3
# ============================================================
# SMART FACE BLUR USING YUNET + DEEPFACE
# ============================================================
"""
High-level Overview:
Program ini mendeteksi wajah secara real-time menggunakan YuNet (OpenCV) dan membandingkan
embedding wajah menggunakan DeepFace (Facenet512). Wajah yang tidak masuk whitelist
akan diblur secara otomatis.

Fitur utama:
1. Deteksi wajah real-time (YuNet)
2. Pengenalan wajah / whitelist (DeepFace)
3. Blur wajah yang tidak dikenal
4. Optimisasi frame-skip untuk mengurangi beban DeepFace
"""

# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================
import os
import cv2
import numpy as np
from deepface import DeepFace
import urllib.request
import math
import time
import tensorflow as tf

# ============================================================
# 1. CPU/GPU CONFIGURATION
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Pakai CPU saja
print("INFO: TensorFlow devices:", tf.config.list_physical_devices())

# ============================================================
# 2. CONFIGURATION PARAMETERS
# ============================================================
SKIP_FRAMES = 30
THRESHOLD = 0.70   # Cosine distance threshold untuk Facenet512
INPUT_RES = (640, 480)  # Resolusi input kamera

fps = 0
prev_time = time.time()

# ============================================================
# 3. YUNET FACE DETECTOR SETUP
# ============================================================
yunet_weights = "weight/face_detection_yunet_2023mar.onnx"
yunet_url = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Download model jika belum ada
if not os.path.exists(yunet_weights):
    print("INFO: Mengunduh model YuNet...")
    try:
        urllib.request.urlretrieve(yunet_url, yunet_weights)
    except Exception as e:
        print(f"ERROR: Download gagal: {e}")
        exit()

backend_id = cv2.dnn.DNN_BACKEND_OPENCV
target_id = cv2.dnn.DNN_TARGET_CPU

face_detector = cv2.FaceDetectorYN.create(
    model=yunet_weights,
    config="",
    input_size=INPUT_RES,
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=backend_id,
    target_id=target_id
)

# ============================================================
# 4. LOAD WHITELIST FACES
# ============================================================
print("INFO: Memuat whitelist...")
whitelist_path = "../whitelist/"
target_embeddings = []

if not os.path.exists(whitelist_path):
    os.makedirs(whitelist_path)

files = [
    os.path.join(whitelist_path, f)
    for f in os.listdir(whitelist_path)
    if os.path.isfile(os.path.join(whitelist_path, f))
]

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

print(f"INFO: Total {len(target_embeddings)} wajah di whitelist.")
target_embeddings = np.array(target_embeddings)

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================
def get_distance(emb1, emb2):
    """Hitung cosine distance antara dua embedding wajah"""
    a = np.matmul(emb1, emb2)
    b = np.linalg.norm(emb1)
    c = np.linalg.norm(emb2)
    return 1 - (a / (b * c))


def is_close(box1, box2, limit=50):
    """Cek apakah posisi bounding box dekat dengan sebelumnya (tracking sederhana)"""
    cx1 = box1[0] + (box1[2] // 2)
    cy1 = box1[1] + (box1[3] // 2)
    cx2 = box2[0] + (box2[2] // 2)
    cy2 = box2[1] + (box2[3] // 2)
    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < limit


def blur_face(face):
    """Blur wajah dengan Gaussian Blur"""
    return cv2.GaussianBlur(face, (51, 51), 30)


# ============================================================
# 6. MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
frame_count = 0
tracked_faces = []

print("INFO: Kamera berjalan. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sframe = cv2.resize(frame, INPUT_RES)

    # ============================================================
    # FPS COUNTER
    # ============================================================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(sframe, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Update ukuran input detector
    h_img, w_img, _ = sframe.shape
    face_detector.setInputSize((w_img, h_img))

    # ============================================================
    # FACE DETECTION
    # ============================================================
    _, faces = face_detector.detect(sframe)
    current_faces_status = []

    if faces is not None:
        for face in faces:
            box = face[0:4].astype(np.int32)
            x, y, w, h = (
                max(0, box[0]),
                max(0, box[1]),
                min(box[2], w_img - box[0]),
                min(box[3], h_img - box[1])
            )
            if w <= 0 or h <= 0:
                continue

            face_img = sframe[y:y+h, x:x+w]
            is_whitelisted = False

            # ============================================================
            # FRAME-SKIP LOGIC (OPTIMIZATION)
            # ============================================================
            needs_recognition = (frame_count % SKIP_FRAMES == 0)
            matched_prev_status = None

            if not needs_recognition:
                for tf in tracked_faces:
                    if is_close(box, tf["box"]):
                        matched_prev_status = tf["status"]
                        break
                if matched_prev_status is not None:
                    is_whitelisted = matched_prev_status
                else:
                    needs_recognition = True

            # ============================================================
            # FACE RECOGNITION
            # ============================================================
            if needs_recognition and len(target_embeddings) > 0:
                try:
                    if face_img.size > 0:
                        # Simpan sementara wajah ke file
                        tmp_path = "../whitelist/tmp_face.jpeg"
                        cv2.imwrite(tmp_path, face_img)

                        curr = DeepFace.represent(
                            img_path=tmp_path,
                            model_name="Facenet512",
                            enforce_detection=False
                        )

                        if curr:
                            curr_emb = curr[0]["embedding"]
                            min_dist = min(
                                get_distance(t, curr_emb)
                                for t in target_embeddings
                            )
                            print(f"DEBUG: Jarak wajah = {min_dist:.4f}")
                            if min_dist <= THRESHOLD:
                                is_whitelisted = True

                except Exception as e:
                    print("ERROR RECOG:", e)

            # ============================================================
            # SAVE TRACKING STATUS
            # ============================================================
            current_faces_status.append({"box": box, "status": is_whitelisted})

            # ============================================================
            # APPLY BLUR
            # ============================================================
            if not is_whitelisted:
                sframe[y:y+h, x:x+w] = blur_face(face_img)

    tracked_faces = current_faces_status
    frame_count += 1
    cv2.imshow("Smart Face Blur", sframe)

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("INFO: Program dihentikan. Kamera ditutup.")