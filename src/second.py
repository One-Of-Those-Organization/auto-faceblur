# #!/usr/bin/env python3
# """
# SISTEM AUTO FACE BLUR - HYBRID ARCHITECTURE (FINAL)
# ---------------------------------------------------
# Detektor    : YOLOv11n (Selalu digunakan untuk mencari kotak wajah)
# Vektorisasi : Selectable (YOLO-Backbone / FaceNet / ArcFace / VGG)
# Fitur       : Auto-Blur Unknown, Real-time FPS, Inference Time Debugging
# """

# import os
# import subprocess
# import time

# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO

# # ============================================================
# # 1. KONFIGURASI SISTEM (SETUP DI SINI)
# # ============================================================

# # PILIH MODEL VEKTORISASI DI SINI:
# # Opsi: "yolo_backbone", "facenet", "arcface", "vgg"
# SELECTED_MODEL = "arcface"

# # Setting Threshold (Batas Kemiripan)
# # Makin kecil angka = Makin ketat (susah dikenali)
# # Makin besar angka = Makin longgar (mudah dikenali)
# THRESHOLDS = {
#     "yolo_backbone": 0.25,  # YOLO butuh angka kecil biar gak salah orang
#     "facenet": 0.38,  # FaceNet stabil di 0.50
#     "arcface": 0.50,  # ArcFace SOTA, 0.50 default aman
#     "vgg": 0.50  # VGG sensitif
# }

# # Path & Tampilan
# SKIP_FRAMES = 1  # Proses setiap frame (1 = Realtime penuh)
# INPUT_RES = (640, 480)  # Resolusi Kamera
# WHITELIST_DIR = "../whitelist/"  # Folder foto wajah sendiri
# PATH_YOLO_MODEL = "../model/model.pt"

# COLOR_KNOWN = (0, 255, 0)  # Hijau (Dikenali)
# COLOR_UNKNOWN = (0, 0, 255)  # Merah (Blur)

# # Ambil threshold sesuai model yang dipilih
# CURRENT_THRESHOLD = THRESHOLDS.get(SELECTED_MODEL, 0.40)

# # Cek Library FaceNet Pytorch
# if SELECTED_MODEL == "facenet":
#     try:
#         from facenet_pytorch import InceptionResnetV1
#     except ImportError:
#         print("Error: Library 'facenet-pytorch' belum terinstall.")
#         print("Run: pip install facenet-pytorch")
#         exit()


# # ============================================================
# # 2. CEK HARDWARE & INISIALISASI
# # ============================================================
# def get_cpu_info():
#     """Mengambil nama processor (Khusus Linux)"""
#     try:
#         return subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | uniq | cut -d: -f2",
#                                        shell=True).decode().strip()
#     except:
#         return "Generic CPU"


# print("\n" + "=" * 50)
# print(f"[*] MODE AKTIF: {SELECTED_MODEL.upper()}")
# print(f"[*] THRESHOLD : {CURRENT_THRESHOLD}")
# print(f"[*] CPU       : {get_cpu_info()}")

# HAS_CUDA = torch.cuda.is_available()
# if HAS_CUDA:
#     print(f"[*] GPU       : {torch.cuda.get_device_name(0)}")
# else:
#     print("[!] WARNING   : Berjalan di CPU (Mungkin lambat)")
# print("=" * 50 + "\n")

# # ============================================================
# # 3. LOAD MODEL (STACKING LOGIC)
# # ============================================================
# print("INFO: Memuat Model ke Memory...")
# try:
#     dev_name = "cuda" if HAS_CUDA else "cpu"
#     device = torch.device(dev_name)

#     # --- A. LOAD DETEKTOR (YOLOv11) ---
#     # Model ini WAJIB ada, apapun metode vektorisasinya
#     model_detector = YOLO(PATH_YOLO_MODEL)
#     model_detector.to(device)
#     print(f"  + Detektor YOLOv11 Loaded di {device}")

#     # --- B. LOAD RECOGNIZER (Sesuai Pilihan) ---
#     model_recognizer = None

#     if SELECTED_MODEL == "facenet":
#         # Load InceptionResnetV1 (Pretrained VGGFace2)
#         model_recognizer = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
#         model_recognizer.eval()  # Mode Evaluasi (Bukan Training)
#         print("  + Recognizer FaceNet Loaded.")

#     elif SELECTED_MODEL == "yolo_backbone":
#         # Tidak perlu load model baru, pakai model_detector yang sudah ada
#         print("  + Recognizer menggunakan Internal Backbone YOLOv11.")

#     elif SELECTED_MODEL in ["arcface", "vgg"]:
#         # DeepFace load modelnya otomatis saat fungsi dipanggil pertama kali
#         # Kita cek librarynya saja
#         try:
#             from deepface import DeepFace

#             print(f"  + Recognizer {SELECTED_MODEL.upper()} via DeepFace Library (Auto-Load).")
#         except ImportError:
#             print("Error: Library 'deepface' belum terinstall.")
#             exit()
#     else:
#         print(f"Error: Model '{SELECTED_MODEL}' tidak dikenal.")
#         exit()

# except Exception as e:
#     print(f"FATAL ERROR saat load model: {e}")
#     exit()


# # ============================================================
# # 4. FUNGSI UTAMA: VEKTORISASI (FEATURE EXTRACTION)
# # ============================================================
# @torch.no_grad()
# def get_embedding(img_bgr):
#     """
#     Mengubah gambar wajah (crop) menjadi vektor angka unik.
#     Logika berubah tergantung SELECTED_MODEL.
#     """
#     if img_bgr is None or img_bgr.size == 0: return None

#     # -------------------------------------------
#     # METODE 1: FACENET (FIXED VERSION)
#     # -------------------------------------------
#     if SELECTED_MODEL == "facenet":
#         # 1. Resize ke 160x160 (Standar FaceNet)
#         input_img = cv2.resize(img_bgr, (160, 160))

#         # 2. Convert BGR ke RGB (Penting!)

#         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

#         # 3. Normalisasi Standar (0-1) <-- INI PERBAIKANNYA
#         # Sebelumnya pakai whitening (dikurangi 127.5), itu sensitif.
#         # Pakai ini lebih stabil untuk crop kasar dari YOLO.

#         tensor = torch.from_numpy(input_img).float().to(device)
#         tensor = tensor / 255.0

#         # 4. Susun Tensor (Batch, Channel, Height, Width)

#         tensor = tensor.permute(2, 0, 1).unsqueeze(0)

#         # 5. Inference

#         return model_recognizer(tensor).flatten().cpu().numpy()

#     # -------------------------------------------
#     # METODE 2: YOLO BACKBONE
#     # -------------------------------------------
#     elif SELECTED_MODEL == "yolo_backbone":
#         # 1. Resize 128x128
#         input_img = cv2.resize(img_bgr, (128, 128))
#         # 2. Normalisasi & Tensor
#         tensor = torch.from_numpy(input_img).float().to(device) / 255.0
#         tensor = tensor.permute(2, 0, 1).unsqueeze(0)

#         # 3. Ambil Output Layer 9 (SPPF)

#         x = tensor
#         for i, layer in enumerate(model_detector.model.model):
#             x = layer(x)
#             if i == 9: break

#         # 4. Flattening

#         emb = torch.mean(x, dim=(2, 3)).flatten()
#         # 5. L2 Norm (Wajib untuk Cosine Distance)
#         return (emb / emb.norm()).cpu().numpy()

#     # -------------------------------------------
#     # METODE 3 & 4: DEEPFACE (ARCFACE / VGG)
#     # -------------------------------------------
#     elif SELECTED_MODEL in ["arcface", "vgg"]:
#         from deepface import DeepFace

#         # DeepFace butuh RGB

#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#         # Mapping nama model

#         model_map = {"arcface": "ArcFace", "vgg": "VGG-Face"}

#         try:
#             # detector_backend="skip" karena kita sudah crop wajah pake YOLO
#             # enforce_detection=False biar gak error kalau wajahnya miring
#             result = DeepFace.represent(
#                 img_path=img_rgb,
#                 model_name=model_map[SELECTED_MODEL],
#                 enforce_detection=False,
#                 detector_backend="skip"
#             )
#             return np.array(result[0]["embedding"])
#         except:
#             return None

#     return None


# # ============================================================
# # 5. FUNGSI PENDUKUNG (JARAK & BLUR)
# # ============================================================
# def get_distance(emb1, emb2):
#     """Menghitung Cosine Distance (0.0 = Sama, 1.0 = Beda)"""
#     return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# def blur_face(face_img):
#     """Efek sensor blur"""
#     return cv2.GaussianBlur(face_img, (51, 51), 30)


# # ============================================================
# # 6. LOAD DATABASE (WHITELIST)
# # ============================================================
# target_embeddings = []
# print("\nINFO: Memproses Database Whitelist...")

# if not os.path.exists(WHITELIST_DIR):
#     os.makedirs(WHITELIST_DIR)
#     print("WARNING: Folder whitelist kosong/baru dibuat.")

# for fname in os.listdir(WHITELIST_DIR):
#     path = os.path.join(WHITELIST_DIR, fname)
#     if os.path.isfile(path):
#         img = cv2.imread(path)
#         if img is not None:
#             # Penting: Kita pakai detektor YOLO dulu di foto whitelist
#             # Supaya crop-nya konsisten dengan saat live webcam
#             res = model_detector(img, verbose=False)
#             found = False
#             for r in res:
#                 if len(r.boxes) > 0:
#                     x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
#                     face_crop = img[y1:y2, x1:x2]
#                     emb = get_embedding(face_crop)
#                     if emb is not None:
#                         target_embeddings.append(emb)
#                         print(f"  + Loaded: {fname} (Wajah Terdeteksi)")
#                         found = True
#                     break  # Ambil 1 wajah saja per foto

#             if not found:
#                 # Fallback: Kalau YOLO gagal detect di foto whitelist, coba full image
#                 # (Biasanya ini terjadi kalau fotonya sudah dicrop pas di wajah)
#                 print(f"  ~ Warning: Deteksi gagal di {fname}, mencoba full image...")
#                 emb = get_embedding(img)
#                 if emb is not None:
#                     target_embeddings.append(emb)
#                     print(f"  + Loaded: {fname} (Full Image)")

# target_embeddings = np.array(target_embeddings)
# print(f"INFO: Total {len(target_embeddings)} embedding wajah siap digunakan.\n")

# # ============================================================
# # 7. MAIN LOOP (KAMERA)
# # ============================================================
# cap = cv2.VideoCapture(0)
# # Paksa resolusi agar load konsisten
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# tracked_faces = []  # Opsional: Bisa dikembangkan untuk tracking ID
# prev_time = time.time()

# print("=================================================")
# print(" KONTROL:")
# print(" [t] : Switch CPU <-> GPU (Tunggu sebentar saat switch)")
# print(" [q] : Keluar")
# print("=================================================\n")

# while True:
#     ok, frame = cap.read()
#     if not ok: break

#     sframe = cv2.resize(frame, INPUT_RES)
#     display_frame = sframe.copy()

#     # Hitung FPS
#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
#     prev_time = curr_time

#     # Mulai hitung Inference Time
#     t_start = time.time()

#     # ---------------------------
#     # TAHAP 1: DETEKSI (YOLO)
#     # ---------------------------
#     results = model_detector(sframe, stream=True, verbose=False)

#     current_faces_status = []

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             # Clipping (Jaga-jaga biar gak error crop)

#             h, w = sframe.shape[:2]
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)

#             face_img = sframe[y1:y2, x1:x2]
#             if face_img.size == 0: continue

#             # ---------------------------
#             # TAHAP 2: PENGENALAN (Vektor)
#             # ---------------------------
#             is_whitelisted = False
#             score = 1.0  # 1.0 = Sangat Beda

#             if len(target_embeddings) > 0:
#                 # Panggil fungsi embedding sesuai model yang dipilih
#                 emb = get_embedding(face_img)

#                 if emb is not None:
#                     # Bandingkan dengan semua data di whitelist
#                     dists = [get_distance(t, emb) for t in target_embeddings]
#                     score = min(dists)  # Ambil jarak terdekat

#                     # Keputusan Akhir
#                     if score <= CURRENT_THRESHOLD:
#                         is_whitelisted = True

#             current_faces_status.append({
#                 'box': [x1, y1, x2, y2],
#                 'status': is_whitelisted,
#                 'score': score
#             })

#     # Selesai hitung Inference Time (Deteksi + Recog)
#     t_end = time.time()
#     inference_ms = (t_end - t_start) * 1000

#     # ---------------------------
#     # TAHAP 3: VISUALISASI
#     # ---------------------------
#     for face in current_faces_status:
#         x1, y1, x2, y2 = face['box']

#         if face['status']:
#             # WHITELIST (Hijau - Jelas)
#             color = COLOR_KNOWN
#             label = f"Me ({face['score']:.2f})"
#         else:
#             # UNKNOWN (Merah - Blur)
#             color = COLOR_UNKNOWN
#             label = f"Unknown ({face['score']:.2f})"

#             # Lakukan Blur di area wajah

#             try:
#                 display_frame[y1:y2, x1:x2] = blur_face(display_frame[y1:y2, x1:x2])
#             except:
#                 pass

#         # Gambar Kotak
#         cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
#         # Gambar Label Background
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#         cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
#         # Tulis Text
#         cv2.putText(display_frame, label, (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#     # ---------------------------
#     # UI INFORMASI
#     # ---------------------------
#     dev_str = "GPU" if "cuda" in str(device) else "CPU"
#     info_str = f"[{dev_str}] {SELECTED_MODEL.upper()} | Infer: {inference_ms:.1f}ms | FPS: {fps:.1f}"

#     cv2.rectangle(display_frame, (0, 0), (640, 35), (0, 0, 0), -1)
#     cv2.putText(display_frame, info_str, (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#     cv2.imshow("Auto Face Blur System", display_frame)

#     # ---------------------------
#     # INPUT KEYBOARD
#     # ---------------------------
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('t'):
#         # Fitur Toggle CPU/GPU
#         print("\n--- Switching Device... ---")
#         new_dev = "cpu" if "cuda" in str(device) else ("cuda" if HAS_CUDA else "cpu")
#         device = torch.device(new_dev)

#         # Pindahkan Model Utama
#         model_detector.to(device)

#         # Pindahkan Model Recognizer (Jika pakai Pytorch)
#         if SELECTED_MODEL == "facenet" and model_recognizer:
#             model_recognizer.to(device)

#         print(f"--- Switched to {new_dev.upper()} ---")

# cap.release()
# cv2.destroyAllWindows()

#!/usr/bin/env python3
"""
SISTEM AUTO FACE BLUR - HYBRID ARCHITECTURE (FINAL + MOBILEFACENET FIX)
---------------------------------------------------
Detektor    : YOLOv11n (Selalu digunakan untuk mencari kotak wajah)
Vektorisasi : Selectable (YOLO-Backbone / FaceNet / ArcFace / VGG / MobileFaceNet)
Fitur       : Auto-Blur Unknown, Real-time FPS, Inference Time Debugging
"""

import os
import time
import cv2
import numpy as np
import torch
import platform  # <--- Ditambah untuk fix error 'cat' di Windows
from ultralytics import YOLO

# ============================================================
# 1. KONFIGURASI SISTEM (SETUP DI SINI)
# ============================================================

# PILIH MODEL VEKTORISASI DI SINI:
# Opsi: "yolo_backbone", "facenet", "arcface", "vgg", "mobilefacenet"
SELECTED_MODEL = "mobilefacenet"

# Config Path Tambahan
PATH_MFNET_CKPT = "../model/068.ckpt"

# Setting Threshold
THRESHOLDS = {
    "yolo_backbone": 0.35,
    "facenet": 0.38,
    "arcface": 0.50,
    "vgg": 0.40,
    "mobilefacenet": 0.45
}

# Path & Tampilan
SKIP_FRAMES = 1
INPUT_RES = (640, 480)
WHITELIST_DIR = "../whitelist/"
PATH_YOLO_MODEL = "../model/model.pt"

COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)

# Ambil threshold sesuai model yang dipilih
CURRENT_THRESHOLD = THRESHOLDS.get(SELECTED_MODEL, 0.40)

# Cek Library FaceNet Pytorch (Hanya jika dipakai)
if SELECTED_MODEL == "facenet":
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print("Error: Library 'facenet-pytorch' belum terinstall.")
        exit()


# ============================================================
# 2. CEK HARDWARE & INISIALISASI
# ============================================================
def get_cpu_info():
    """Mengambil nama processor (Cross-Platform)"""
    # Fix: Pakai platform.processor() biar aman di Windows & Linux
    return platform.processor()

print("\n" + "=" * 50)
print(f"[*] MODE AKTIF: {SELECTED_MODEL.upper()}")
print(f"[*] THRESHOLD : {CURRENT_THRESHOLD}")
print(f"[*] CPU       : {get_cpu_info()}")

HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    print(f"[*] GPU       : {torch.cuda.get_device_name(0)}")
else:
    print("[!] WARNING   : Berjalan di CPU (Mungkin lambat)")
print("=" * 50 + "\n")

# ============================================================
# 3. LOAD MODEL (STACKING LOGIC)
# ============================================================
print("INFO: Memuat Model ke Memory...")
try:
    dev_name = "cuda" if HAS_CUDA else "cpu"
    device = torch.device(dev_name)

    # --- A. LOAD DETEKTOR (YOLOv11) ---
    model_detector = YOLO(PATH_YOLO_MODEL)
    model_detector.to(device)
    print(f"  + Detektor YOLOv11 Loaded di {device}")

    # --- B. LOAD RECOGNIZER (Sesuai Pilihan) ---
    model_recognizer = None

    if SELECTED_MODEL == "facenet":
        model_recognizer = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
        model_recognizer.eval()
        print("  + Recognizer FaceNet Loaded.")

    elif SELECTED_MODEL == "yolo_backbone":
        print("  + Recognizer menggunakan Internal Backbone YOLOv11.")

    elif SELECTED_MODEL in ["arcface", "vgg"]:
        try:
            from deepface import DeepFace
            print(f"  + Recognizer {SELECTED_MODEL.upper()} via DeepFace Library (Auto-Load).")
        except ImportError:
            print("Error: Library 'deepface' belum terinstall.")
            exit()
            
    # --- METODE: MOBILEFACENET ---
    elif SELECTED_MODEL == "mobilefacenet":
        try:
            from mfnet import MobileFacenet 
            model_recognizer = MobileFacenet()
            
            # Load Checkpoint Manual
            if os.path.exists(PATH_MFNET_CKPT):
                # Load map_location='cpu' dulu biar aman
                checkpoint = torch.load(PATH_MFNET_CKPT, map_location='cpu')
                if 'net_state_dict' in checkpoint:
                    model_recognizer.load_state_dict(checkpoint['net_state_dict'], strict=False)
                else:
                    model_recognizer.load_state_dict(checkpoint, strict=False)
            else:
                print(f"ERR: File checkpoint '{PATH_MFNET_CKPT}' tidak ditemukan!")
                exit()
                
            model_recognizer.to(device)
            model_recognizer.eval()
            print("  + Recognizer MobileFaceNet Loaded.")
        except ImportError:
            print("ERR: File 'mfnet.py' tidak ditemukan. MobileFaceNet butuh file ini.")
            exit()
        except Exception as e:
            print(f"ERR: Gagal load MobileFaceNet: {e}")
            exit()
            
    else:
        print(f"Error: Model '{SELECTED_MODEL}' tidak dikenal.")
        exit()

except Exception as e:
    print(f"FATAL ERROR saat load model: {e}")
    exit()


# ============================================================
# 4. FUNGSI UTAMA: VEKTORISASI (FEATURE EXTRACTION)
# ============================================================
@torch.no_grad()
def get_embedding(img_bgr):
    """
    Mengubah gambar wajah (crop) menjadi vektor angka unik.
    Logika berubah tergantung SELECTED_MODEL.
    """
    if img_bgr is None or img_bgr.size == 0: return None

    # -------------------------------------------
    # METODE 1: FACENET
    # -------------------------------------------
    if SELECTED_MODEL == "facenet":
        input_img = cv2.resize(img_bgr, (160, 160))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(input_img).float().to(device)
        tensor = tensor / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return model_recognizer(tensor).flatten().cpu().numpy()

    # -------------------------------------------
    # METODE 2: YOLO BACKBONE
    # -------------------------------------------
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

    # -------------------------------------------
    # METODE 3 & 4: DEEPFACE (ARCFACE / VGG)
    # -------------------------------------------
    elif SELECTED_MODEL in ["arcface", "vgg"]:
        from deepface import DeepFace
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        model_map = {"arcface": "ArcFace", "vgg": "VGG-Face"}
        try:
            result = DeepFace.represent(
                img_path=img_rgb,
                model_name=model_map[SELECTED_MODEL],
                enforce_detection=False,
                detector_backend="skip"
            )
            return np.array(result[0]["embedding"])
        except:
            return None

    # -------------------------------------------
    # METODE 5: MOBILEFACENET (FIXED UNPACKING ERROR)
    # -------------------------------------------
    elif SELECTED_MODEL == "mobilefacenet":
        img = cv2.resize(img_bgr, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        
        # --- PERBAIKAN LOGIKA DISINI ---
        # MobileFaceNet bisa return tuple atau tensor doang
        # Kita handle dua-duanya biar gak error value unpack
        output = model_recognizer(img)
        
        if isinstance(output, tuple):
            emb = output[0]
        else:
            emb = output
        
        emb = emb.flatten().cpu().numpy()
        emb = emb / np.linalg.norm(emb) # L2 Norm
        return emb

    return None


# ============================================================
# 5. FUNGSI PENDUKUNG (JARAK & BLUR)
# ============================================================
def get_distance(emb1, emb2):
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def blur_face(face_img):
    return cv2.GaussianBlur(face_img, (51, 51), 30)


# ============================================================
# 6. LOAD DATABASE (WHITELIST)
# ============================================================
target_embeddings = []
print("\nINFO: Memproses Database Whitelist...")

if not os.path.exists(WHITELIST_DIR):
    os.makedirs(WHITELIST_DIR)
    print("WARNING: Folder whitelist kosong/baru dibuat.")

for fname in os.listdir(WHITELIST_DIR):
    path = os.path.join(WHITELIST_DIR, fname)
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            res = model_detector(img, verbose=False)
            found = False
            for r in res:
                if len(r.boxes) > 0:
                    x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
                    face_crop = img[y1:y2, x1:x2]
                    emb = get_embedding(face_crop)
                    if emb is not None:
                        target_embeddings.append(emb)
                        print(f"  + Loaded: {fname} (Wajah Terdeteksi)")
                        found = True
                    break

            if not found:
                print(f"  ~ Warning: Deteksi gagal di {fname}, mencoba full image...")
                emb = get_embedding(img)
                if emb is not None:
                    target_embeddings.append(emb)
                    print(f"  + Loaded: {fname} (Full Image)")

target_embeddings = np.array(target_embeddings)
print(f"INFO: Total {len(target_embeddings)} embedding wajah siap digunakan.\n")

# ============================================================
# 7. MAIN LOOP (KAMERA)
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracked_faces = []
prev_time = time.time()

print("=================================================")
print(" KONTROL:")
print(" [t] : Switch CPU <-> GPU (Tunggu sebentar saat switch)")
print(" [q] : Keluar")
print("=================================================\n")

while True:
    ok, frame = cap.read()
    if not ok: break

    sframe = cv2.resize(frame, INPUT_RES)
    display_frame = sframe.copy()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Inference Time
    t_start = time.time()

    # DETEKSI
    results = model_detector(sframe, stream=True, verbose=False)
    current_faces_status = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = sframe.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = sframe[y1:y2, x1:x2]
            if face_img.size == 0: continue

            # PENGENALAN
            is_whitelisted = False
            score = 1.0

            if len(target_embeddings) > 0:
                emb = get_embedding(face_img)
                if emb is not None:
                    dists = [get_distance(t, emb) for t in target_embeddings]
                    score = min(dists)
                    if score <= CURRENT_THRESHOLD:
                        is_whitelisted = True

            current_faces_status.append({
                'box': [x1, y1, x2, y2],
                'status': is_whitelisted,
                'score': score
            })

    t_end = time.time()
    inference_ms = (t_end - t_start) * 1000

    # VISUALISASI
    for face in current_faces_status:
        x1, y1, x2, y2 = face['box']
        
        if face['status']:
            color = COLOR_KNOWN
            label = f"Me ({face['score']:.2f})"
        else:
            color = COLOR_UNKNOWN
            label = f"Unknown ({face['score']:.2f})"
            try:
                display_frame[y1:y2, x1:x2] = blur_face(display_frame[y1:y2, x1:x2])
            except: pass

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # UI INFO
    dev_str = "GPU" if "cuda" in str(device) else "CPU"
    info_str = f"[{dev_str}] {SELECTED_MODEL.upper()} | Infer: {inference_ms:.1f}ms | FPS: {fps:.1f}"
    
    cv2.rectangle(display_frame, (0, 0), (640, 35), (0, 0, 0), -1)
    cv2.putText(display_frame, info_str, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Auto Face Blur System", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        print("\n--- Switching Device... ---")
        new_dev = "cpu" if "cuda" in str(device) else ("cuda" if HAS_CUDA else "cpu")
        device = torch.device(new_dev)
        
        model_detector.to(device)
        
        if SELECTED_MODEL == "facenet" and model_recognizer:
            model_recognizer.to(device)
        elif SELECTED_MODEL == "mobilefacenet" and model_recognizer:
            model_recognizer.to(device)
            
        print(f"--- Switched to {new_dev.upper()} ---")

cap.release()
cv2.destroyAllWindows()