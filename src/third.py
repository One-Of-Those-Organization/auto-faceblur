#!/usr/bin/env python3
"""
Auto Face Blur using YOLOv11 Embeddings

Fungsi:
- Deteksi wajah real-time dengan YOLOv11
- Membuat embedding wajah untuk whitelist
- Blur wajah yang tidak ada di whitelist
- Tracking sederhana untuk efisiensi
"""

import os
import cv2
import numpy as np
import math
import time
import torch
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
SKIP_FRAMES = 20          # Interval frame untuk melakukan recognition
THRESHOLD = 0.45          # Cosine distance threshold untuk whitelist
INPUT_RES = (640, 480)    # Resolusi input kamera

fps = 0
prev_time = time.time()

# ============================================================
# LOAD YOLO MODEL
# ============================================================
MODEL_PATH = "../model/model.pt"  # Pastikan path ini sesuai lokasi file
print("INFO: Loading YOLOv11 model...")
model = YOLO(MODEL_PATH)

# Tentukan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("INFO: Model loaded on", device)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_distance(emb1, emb2):
    """Hitung cosine distance antara dua embedding"""
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

@torch.no_grad()
def get_embedding(img_bgr):
    """Ambil embedding wajah dari YOLO backbone"""
    if img_bgr is None or img_bgr.size == 0:
        return None

    # Resize ke 128x128 untuk konsistensi
    input_img = cv2.resize(img_bgr, (128, 128))
    tensor = torch.from_numpy(input_img).float().to(device)
    tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (B, C, H, W)

    # Ambil fitur dari backbone YOLO (layer 0-9)
    x = tensor
    for i, layer in enumerate(model.model.model):
        x = layer(x)
        if i == 9:
            break

    # Global Average Pooling + L2 Normalization
    embedding = torch.mean(x, dim=(2, 3)).flatten()
    embedding = embedding / embedding.norm()
    return embedding.cpu().numpy()

def blur_face(face):
    """Gaussian blur untuk wajah"""
    return cv2.GaussianBlur(face, (51, 51), 30)

def is_close(box1, box2, limit=50):
    """Cek apakah dua bounding box dekat (tracking sederhana)"""
    cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    cx2, cy2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return dist < limit

# ============================================================
# LOAD WHITELIST EMBEDDINGS
# ============================================================
WHITELIST_PATH = "../whitelist/"
os.makedirs(WHITELIST_PATH, exist_ok=True)

target_embeddings = []
for fname in os.listdir(WHITELIST_PATH):
    path = os.path.join(WHITELIST_PATH, fname)
    if not os.path.isfile(path):
        continue
    img = cv2.imread(path)
    if img is None:
        print("  - Failed to load", fname)
        continue
    emb = get_embedding(img)
    if emb is not None:
        target_embeddings.append(emb)
        print("  + Loaded:", fname)
    else:
        print("  - No face in", fname)

target_embeddings = np.array(target_embeddings)
print(f"INFO: {len(target_embeddings)} whitelist embeddings loaded.")

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
frame_count = 0
tracked_faces = []

print("INFO: Camera running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sframe = cv2.resize(frame, INPUT_RES)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(sframe, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Deteksi wajah
    results = model(sframe, stream=True, verbose=False)
    current_faces_status = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = sframe.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = sframe[y1:y2, x1:x2]
            if face is None or face.size == 0:
                continue

            # Default status
            is_whitelisted = False
            needs_recognition = (frame_count % SKIP_FRAMES == 0)

            # Tracking sederhana
            if not needs_recognition:
                for tf in tracked_faces:
                    if is_close([x1,y1,x2,y2], tf["box"]):
                        is_whitelisted = tf["status"]
                        break
                else:
                    needs_recognition = True

            # Recognition
            if needs_recognition and len(target_embeddings) > 0:
                emb = get_embedding(face)
                if emb is not None:
                    dists = [get_distance(t, emb) for t in target_embeddings]
                    if min(dists) <= THRESHOLD:
                        is_whitelisted = True

            current_faces_status.append({
                "box": [x1, y1, x2, y2],
                "status": is_whitelisted
            })

            # Apply blur
            if not is_whitelisted:
                sframe[y1:y2, x1:x2] = blur_face(face)

    tracked_faces = current_faces_status
    frame_count += 1

    cv2.imshow("Auto Face Blur", sframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()