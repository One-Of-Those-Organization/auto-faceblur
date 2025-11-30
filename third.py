#!/usr/bin/env python2
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
SKIP_FRAMES = 20          # perform recognition every N frames
THRESHOLD = 0.45          # cosine distance threshold for whitelist match
INPUT_RES = (640, 480)

fps = 0
prev_time = time.time()

# ============================================================
# LOAD YOLOv11 FACE MODEL
# ============================================================
print("INFO: Loading YOLOv11 model...")
model = YOLO("model.pt")    # your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("INFO: YOLO model loaded on", device)

# ============================================================
# COSINE DISTANCE
# ============================================================
def get_distance(emb1, emb2):
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# ============================================================
# YOLO BACKBONE EMBEDDING FUNCTION
# ============================================================
# Uses first 12 backbone layers → global average pooling → normalized embedding
# Lightweight and fast (1–3 ms)
# ============================================================
# YOLO BACKBONE EMBEDDING FUNCTION
# ============================================================
@torch.no_grad()
def get_yolo_embedding(img_bgr):
    # 1. Safety check
    if img_bgr is None or img_bgr.size == 0:
        return None

    # 2. Preprocessing
    # Resize to a fixed input size (critical for consistent embeddings)
    # 128x128 is a good balance between speed and feature retention for faces
    input_img = cv2.resize(img_bgr, (128, 128))

    # Convert BGR to RGB, normalize to 0-1, and change layout to (B, C, H, W)
    tensor = torch.from_numpy(input_img).to(device).float()
    tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) # Add batch dimension

    # 3. Feature Extraction (Backbone only)
    # Access the internal PyTorch model.
    # In YOLOv8/v11, the backbone usually ends at layer 9 (SPPF layer).
    # We iterate through the first 10 layers (0-9).
    x = tensor
    # model.model is the DetectionModel, model.model.model is the nn.Sequential
    for i, layer in enumerate(model.model.model):
        x = layer(x)
        if i == 9: # Stop after the SPPF layer (end of backbone)
            break

    # x is now a feature map (e.g., [1, 512, 4, 4])

    # 4. Global Average Pooling (GAP)
    # Squash spatial dimensions (H, W) to create a single vector
    embedding = torch.mean(x, dim=(2, 3)).flatten()

    # 5. L2 Normalization
    # Essential for Cosine Similarity to work correctly
    embedding = embedding / embedding.norm()

    return embedding.cpu().numpy()


# ============================================================
# LOAD WHITELIST EMBEDDINGS
# ============================================================
print("INFO: Loading whitelist...")
whitelist_path = "whitelist/"
target_embeddings = []

os.makedirs(whitelist_path, exist_ok=True)

for fname in os.listdir(whitelist_path):
    path = os.path.join(whitelist_path, fname)
    if not os.path.isfile(path):
        continue

    img = cv2.imread(path)
    if img is None:
        print("  - Failed to load", fname)
        continue

    emb = get_yolo_embedding(img)
    if emb is not None:
        target_embeddings.append(emb)
        print("  + Loaded:", fname)
    else:
        print("  - No face in", fname)

target_embeddings = np.array(target_embeddings)
print("INFO:", len(target_embeddings), "whitelist embeddings loaded.")

# ============================================================
# HELPERS
# ============================================================
def blur_face(face):
    return cv2.GaussianBlur(face, (51, 51), 30)

def is_close(box1, box2, limit=50):
    cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    cx2, cy2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return dist < limit

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
frame_count = 0
tracked_faces = []

print("INFO: Camera running... Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    sframe = cv2.resize(frame, INPUT_RES)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(sframe, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # YOLO detection
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

            is_whitelisted = False
            needs_recognition = (frame_count % SKIP_FRAMES == 0)

            # Tracking
            if not needs_recognition:
                for tf in tracked_faces:
                    if is_close([x1,y1,x2,y2], tf["box"]):
                        is_whitelisted = tf["status"]
                        break
                else:
                    needs_recognition = True

            # Recognition
            if needs_recognition and len(target_embeddings) > 0:
                emb = get_yolo_embedding(face)
                if emb is not None:
                    dists = [get_distance(t, emb) for t in target_embeddings]
                    if min(dists) <= THRESHOLD:
                        is_whitelisted = True

            current_faces_status.append({
                "box": [x1, y1, x2, y2],
                "status": is_whitelisted
            })

            # Blur or highlight
            if not is_whitelisted:
                sframe[y1:y2, x1:x2] = blur_face(face)
            else:
                cv2.rectangle(sframe, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

    tracked_faces = current_faces_status
    frame_count += 1

    cv2.imshow("Auto Face Blur (YOLO Embedding)", sframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
