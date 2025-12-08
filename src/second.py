#!/usr/bin/env python3
"""
Auto Face Blur - Multi-Model Architecture
Fitur:
1. Deteksi: SELALU YOLOv11
2. Vektorisasi: Bisa pilih banyak (Stacked: YOLO, FaceNet, ArcFace, VGG)
3. Toggle CPU/GPU
"""

import os
import subprocess
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# 1. KONFIGURASI MODEL (PILIH SALAH SATU)
# ============================================================
# Pilihan: "yolo_backbone", "facenet", "arcface", "vgg"

SELECTED_MODEL = "facenet"  # <--- Sedang pakai ARCFACE

# Config Lain
SKIP_FRAMES = 1
THRESHOLD = 0.50
INPUT_RES = (640, 480)
WHITELIST_DIR = "../whitelist/"
COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)


# ============================================================
# 2. CEK HARDWARE
# ============================================================
def get_cpu_name():
    try:
        return subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | uniq | cut -d: -f2",
                                       shell=True).decode().strip()
    except:
        return "Unknown CPU"


HAS_CUDA = torch.cuda.is_available()
print(f"[*] Mode: {SELECTED_MODEL.upper()}")
print(f"[*] CPU : {get_cpu_name()}")
print(f"[*] GPU : {torch.cuda.get_device_name(0) if HAS_CUDA else 'None'}")

# ============================================================
# 3. LOAD MODEL (STACKING AREA)
# ============================================================
print("\nINFO: Memuat Model...")
try:
    dev_name = "cuda" if HAS_CUDA else "cpu"
    device = torch.device(dev_name)

    # --- A. MODEL DETEKSI (TETAP) ---
    model_detector = YOLO("../model/model.pt")
    model_detector.to(device)
    print(f"  + Detektor YOLOv11 Loaded")

    # --- B. MODEL VEKTORISASI (STACKED) ---
    model_recognizer = None

    # [OPSI 1] FaceNet (Pytorch)
    if SELECTED_MODEL == "facenet":
        try:
            from facenet_pytorch import InceptionResnetV1

            model_recognizer = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
            model_recognizer.eval()
            print("  + Recognizer: FaceNet Loaded")
        except ImportError:
            print("ERR: Install facenet-pytorch dulu")
            exit()

    # [OPSI 2] YOLO Backbone
    elif SELECTED_MODEL == "yolo_backbone":
        print("  + Recognizer: Using YOLOv11 Internal Backbone")

    # [OPSI 3 & 4] ArcFace & VGG-Face (Via DeepFace)
    elif SELECTED_MODEL in ["arcface", "vgg"]:
        try:
            # !!! INI YANG TADI KURANG !!!
            from deepface import DeepFace

            # Kita panggil sekali biar dia download weight di awal
            print(f"  + Recognizer: {SELECTED_MODEL.upper()} (via DeepFace) Initializing...")
            print("    (Pertama kali akan download model, tunggu sebentar...)")

            # Dummy run biar model ke-load ke memory
            # DeepFace modelnya diload otomatis saat pemanggilan fungsi represent
        except ImportError:
            print("ERR: Library deepface belum ada.")
            print("Run: pip install deepface tf-keras")
            exit()

    else:
        print(f"Error: Model '{SELECTED_MODEL}' tidak dikenali!")
        exit()

except Exception as e:
    print(f"ERROR Load Model: {e}")
    exit()


# ============================================================
# 4. FUNGSI EMBEDDING (STACKING AREA)
# ============================================================
@torch.no_grad()
def get_embedding(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return None

    # -----------------------------------------------------
    # METODE 1: FACENET
    # -----------------------------------------------------
    if SELECTED_MODEL == "facenet":
        input_img = cv2.resize(img_bgr, (160, 160))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.float32(input_img)
        input_img = (input_img - 127.5) / 128.0
        tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)
        return model_recognizer(tensor).flatten().cpu().numpy()

    # -----------------------------------------------------
    # METODE 2: YOLO BACKBONE
    # -----------------------------------------------------
    elif SELECTED_MODEL == "yolo_backbone":
        input_img = cv2.resize(img_bgr, (128, 128))
        tensor = torch.from_numpy(input_img).float().to(device) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        x = tensor
        for i, layer in enumerate(model_detector.model.model):
            x = layer(x)
            if i == 9: break
        emb = torch.mean(x, dim=(2, 3)).flatten()
        return (emb / emb.norm()).cpu().numpy()

    # -----------------------------------------------------
    # METODE 3 & 4: DEEPFACE (ARCFACE / VGG)
    # -----------------------------------------------------
    elif SELECTED_MODEL in ["arcface", "vgg"]:
        # DeepFace butuh import di scope ini juga atau global
        from deepface import DeepFace

        # DeepFace butuh RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Nama model sesuai config string DeepFace
        model_name_map = {
            "arcface": "ArcFace",
            "vgg": "VGG-Face"
        }

        try:
            # enforce_detection=False karena wajah sudah dicari oleh YOLO
            results = DeepFace.represent(
                img_path=img_rgb,
                model_name=model_name_map[SELECTED_MODEL],
                enforce_detection=False,
                detector_backend="skip"
            )
            return np.array(results[0]["embedding"])
        except:
            return None

    return None


# ============================================================
# 5. CORE LOGIC
# ============================================================
def get_distance(emb1, emb2):
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def blur_face(face_img):
    return cv2.GaussianBlur(face_img, (51, 51), 30)


# Generate Whitelist
target_embeddings = []
print("INFO: Generating Whitelist...")
if not os.path.exists(WHITELIST_DIR): os.makedirs(WHITELIST_DIR)
for fname in os.listdir(WHITELIST_DIR):
    path = os.path.join(WHITELIST_DIR, fname)
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            emb = get_embedding(img)
            if emb is not None:
                target_embeddings.append(emb)
                print(f"  + Loaded: {fname}")
target_embeddings = np.array(target_embeddings)
print(f"INFO: {len(target_embeddings)} wajah siap.")

# ============================================================
# 6. MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
# Paksa resolusi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracked_faces = []
prev_time = time.time()
print("\nControls: [t] Switch CPU/GPU | [q] Quit\n")

while True:
    ok, frame = cap.read()
    if not ok: break

    sframe = cv2.resize(frame, INPUT_RES)
    display_frame = sframe.copy()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Inference Time Start
    t_start = time.time()

    # 1. DETEKSI (YOLO)
    results = model_detector(sframe, stream=True, verbose=False)

    status_list = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Clipping
            h, w = sframe.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = sframe[y1:y2, x1:x2]
            if face.size == 0: continue

            # 2. RECOGNITION (Sesuai SELECTED_MODEL)
            is_white = False
            score = 1.0

            if len(target_embeddings) > 0:
                emb = get_embedding(face)
                if emb is not None:
                    dists = [get_distance(t, emb) for t in target_embeddings]
                    score = min(dists)
                    if score <= THRESHOLD: is_white = True

            status_list.append({'box': [x1, y1, x2, y2], 'ok': is_white, 'sc': score})

    # Inference Time End
    inference_ms = (time.time() - t_start) * 1000

    # VISUALISASI
    for item in status_list:
        x1, y1, x2, y2 = item['box']
        color = COLOR_KNOWN if item['ok'] else COLOR_UNKNOWN
        label = f"{'Me' if item['ok'] else 'Unknown'} ({item['sc']:.2f})"

        if not item['ok']:
            try:
                display_frame[y1:y2, x1:x2] = blur_face(display_frame[y1:y2, x1:x2])
            except:
                pass

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # INFO BAR
    info = f"[{'GPU' if 'cuda' in str(device) else 'CPU'}] {SELECTED_MODEL.upper()} | {inference_ms:.1f}ms"
    cv2.putText(display_frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Multi-Model Benchmark", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        new_dev = "cpu" if "cuda" in str(device) else ("cuda" if HAS_CUDA else "cpu")
        device = torch.device(new_dev)

        # Pindah YOLO
        model_detector.to(device)

        # Pindah FaceNet (Kalau aktif)
        if SELECTED_MODEL == "facenet" and model_recognizer:
            model_recognizer.to(device)

        # DeepFace (ArcFace/VGG) otomatis handle device via TensorFlow/Keras environment,
        # jadi ga perlu manual .to(device) seperti PyTorch

        print(f"SWITCH -> {new_dev.upper()}")

cap.release()
cv2.destroyAllWindows()
