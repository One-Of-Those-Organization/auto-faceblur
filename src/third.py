#!/usr/bin/env python3
"""
Auto Face Blur using YOLOv11 Backbone Embeddings

- Detect faces in real-time using YOLOv11
- Generate embeddings from YOLO backbone
- Compare with whitelist embeddings
- Blur faces not in whitelist
"""

import os
import cv2
import numpy as np
import math
import time
import torch
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
SKIP_FRAMES = 5           # Recognition every N frames
THRESHOLD = 0.35          # Cosine distance threshold for whitelist match
INPUT_RES = (640, 480)    # Camera input resolution

fps = 0
prev_time = time.time()

# ============================================================
# LOAD MODEL
# ============================================================
print("INFO: Loading YOLOv11 model...")
model = YOLO("../model/model.pt")  # Update path if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("INFO: Model loaded on", device)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_distance(emb1, emb2):
    n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return 1 - np.dot(emb1, emb2) / (n1 * n2)

@torch.no_grad()
def get_embedding(img_bgr):
    """Generate normalized embedding from YOLO backbone features."""
    if img_bgr is None or img_bgr.size == 0:
        return None

    # Preprocess: resize and normalize
    input_img = cv2.resize(img_bgr, (128, 128))
    tensor = torch.from_numpy(input_img).float().to(device) / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # B, C, H, W

    # Extract features from YOLO backbone (first 10 layers)
    x = tensor
    for i, layer in enumerate(model.model.model):
        x = layer(x)
        if i == 9:
            break

    # Global average pooling + L2 normalization
    embedding = torch.mean(x, dim=(2, 3)).flatten()
    embedding = embedding / embedding.norm()

    return embedding.cpu().numpy()


def blur_face(face):
    """Apply Gaussian blur to face region."""
    return cv2.GaussianBlur(face, (51, 51), 30)

def is_close(box1, box2, limit=50):
    """Check if two bounding boxes are spatially close (for tracking)."""
    cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    cx2, cy2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < limit

# ============================================================
# LOAD WHITELIST EMBEDDINGS
# ============================================================
print("INFO: Loading whitelist...")
whitelist_path = "../whitelist/"
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
    ok, frame = cap.read()
    if not ok:
        break

    sframe = cv2.resize(frame, INPUT_RES)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(sframe, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Detect faces using YOLO
    results = model(sframe, stream=True, verbose=False)
    current_faces_status = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # clamp to frame
            h, w, _ = sframe.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face = sframe[y1:y2, x1:x2]
            if face is None or face.size == 0:
                continue

            is_whitelisted = False
            needs_recognition = (frame_count % SKIP_FRAMES == 0)
            dist = None  # store last distance

            # Tracking previous faces
            if not needs_recognition:
                for tf in tracked_faces:
                    if is_close([x1, y1, x2, y2], tf["box"]):
                        is_whitelisted = tf["status"]
                        dist = tf.get("dist")
                        break
                else:
                    needs_recognition = True

            # Recognition using embeddings
            if needs_recognition and len(target_embeddings) > 0:
                emb = get_embedding(face)
                if emb is not None:
                    dists = [get_distance(t, emb) for t in target_embeddings]
                    best = float(min(dists))
                    if best <= THRESHOLD:
                        is_whitelisted = True
                        dist = best

            current_faces_status.append({
                "box": [x1, y1, x2, y2],
                "status": is_whitelisted,
                "dist": dist,
            })

            # Apply blur if not whitelisted
            if not is_whitelisted:
                sframe[y1:y2, x1:x2] = blur_face(face)
            else:
                shown = dist if dist is not None else 0.0
                cv2.putText(sframe, f"Dist: {shown:.3f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    tracked_faces = current_faces_status
    frame_count += 1

    cv2.imshow("Auto Face Blur", sframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
