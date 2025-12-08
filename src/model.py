#!/usr/bin/env python3
"""
Source: second.py
"""

import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------
class ActiveModelConfig:
    def __init__(
        self,
        selected_model="facenet",
        skip_frames=1,
        threshold=0.5,
        res=(640, 480),
        whitelist_dir="../whitelist/"
    ):
        self.selected_model = selected_model
        self.skip_frames = skip_frames
        self.threshold = threshold
        self.res = res
        self.whitelist_dir = whitelist_dir


# ------------------------------------------------------------
# 2. MAIN CLASS
# ------------------------------------------------------------
class ActiveModel:
    def __init__(self, config: ActiveModelConfig = ActiveModelConfig()):
        self.cfg = config

        # Device
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.has_cuda else "cpu")

        # Detector
        self.detector = YOLO("./model/model.pt")
        self.detector.to(self.device)
        print("INFO: YOLOv11 loaded")

        # Recognizer (stacking)
        self.recognizer = self._load_recognizer()

        # Whitelist
        self.target_embeddings = self._load_whitelist()

    # ------------------------------------------------------------
    # RECOGNIZER LOADING
    # ------------------------------------------------------------
    def _load_recognizer(self):
        model_name = self.cfg.selected_model

        # 1. FACENET
        if model_name == "facenet":
            try:
                from facenet_pytorch import InceptionResnetV1
                model = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device)
                model.eval()
                print("INFO: FaceNet loaded")
                return model
            except ImportError:
                print("ERR: facenet-pytorch missing")
                return None

        # 2. YOLO backbone = no external model
        if model_name == "yolo_backbone":
            print("INFO: YOLO backbone used for embeddings")
            return None

        # 3. DEEPFACE (ArcFace/VGG)
        if model_name in ["arcface", "vgg"]:
            from deepface import DeepFace
            print(f"INFO: DeepFace {model_name} initialized")
            return True  # Flag only

        return None

    # ------------------------------------------------------------
    # LOAD WHITELIST FACES
    # ------------------------------------------------------------
    def _load_whitelist(self):
        folder = self.cfg.whitelist_dir
        out = []

        if not os.path.exists(folder):
            os.makedirs(folder)

        print("INFO: Generating whitelist...")
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            img = cv2.imread(path)
            if img is None:
                continue

            emb = self.get_embedding(img)
            if emb is not None:
                out.append(emb)
                print(f" - Loaded: {f}")

        print(f"INFO: {len(out)} embeddings loaded.")
        return np.array(out)

    # ------------------------------------------------------------
    # GET EMBEDDING (SEVERAL BACKENDS)
    # ------------------------------------------------------------
    @torch.no_grad()
    def get_embedding(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return None

        model = self.cfg.selected_model

        # --------------------- FACENET ---------------------
        if model == "facenet":
            x = cv2.resize(img_bgr, (160, 160))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = (x - 127.5) / 128.0
            x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            return self.recognizer(x).flatten().cpu().numpy()

        # --------------------- YOLO BACKBONE ---------------------
        if model == "yolo_backbone":
            x = cv2.resize(img_bgr, (128, 128))
            x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

            y = x
            for i, layer in enumerate(self.detector.model.model):
                y = layer(y)
                if i == 9: break

            emb = torch.mean(y, dim=(2, 3)).flatten()
            return (emb / emb.norm()).cpu().numpy()

        # --------------------- DEEPFACE (ARC/VGG) ---------------------
        if model in ["arcface", "vgg"]:
            from deepface import DeepFace
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            model_name_map = {"arcface": "ArcFace", "vgg": "VGG-Face"}

            try:
                res = DeepFace.represent(
                    img_path=rgb,
                    model_name=model_name_map[model],
                    enforce_detection=False,
                    detector_backend="skip"
                )
                return np.array(res[0]["embedding"])
            except:
                return None

        return None

    # ------------------------------------------------------------
    # DISTANCE FUNCTION
    # ------------------------------------------------------------
    def get_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ------------------------------------------------------------
    # BLUR
    # ------------------------------------------------------------
    def blur(self, img):
        return cv2.GaussianBlur(img, (51, 51), 30)

    # ------------------------------------------------------------
    # MAIN PREDICT METHOD (NO WINDOW)
    # ------------------------------------------------------------
    def predict_frame(self, frame):
        """
        Input:
            frame (BGR)

        Output:
            processed_frame, detections

        detections = [
            {
                "box": [x1,y1,x2,y2],
                "ok": True/False,
                "score": float
            }
        ]
        """
        if frame is None:
            return None, []

        sframe = cv2.resize(frame, self.cfg.res)
        proc = sframe.copy()
        detections = []

        # YOLO detect
        results = self.detector(sframe, stream=True, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = sframe.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = sframe[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                is_known = False
                score = 1.0

                if len(self.target_embeddings) > 0:
                    emb = self.get_embedding(face)
                    if emb is not None:
                        dists = [self.get_distance(t, emb) for t in self.target_embeddings]
                        score = min(dists)
                        is_known = score <= self.cfg.threshold

                # Blur unknown
                if not is_known:
                    try:
                        proc[y1:y2, x1:x2] = self.blur(proc[y1:y2, x1:x2])
                    except:
                        pass

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "ok": is_known,
                    "score": score
                })

        return proc, detections
