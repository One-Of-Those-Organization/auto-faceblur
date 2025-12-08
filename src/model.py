#!/usr/bin/env python3
"""
Source: model.py (fixed)
"""

import os
import cv2
import torch
import numpy as np
import deepface
from ultralytics import YOLO


class ActiveModelConfig:
    def __init__(
        self,
        selected_model="yolo_backbone",
        skip_frames=20,
        threshold=0.5,
        res=(640, 480),
        whitelist_dir="../whitelist/"
    ):
        self.selected_model = selected_model
        self.skip_frames = skip_frames
        self.threshold = threshold
        self.res = res
        self.whitelist_dir = whitelist_dir


class ActiveModel:
    def __init__(self, config: ActiveModelConfig = ActiveModelConfig()):
        self.cfg = config

        # Device
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.has_cuda else "cpu")

        # Detector
        try:
            self.detector = YOLO("./model/model.pt")
            self.detector.to(self.device)
            print("INFO: YOLO model loaded")
        except Exception as e:
            print(f"ERR: Failed to load YOLO model: {e}")
            self.detector = None

        # Recognizer (stacking)
        self.recognizer = self._load_recognizer()

        # Whitelist
        self.target_embeddings = self._load_whitelist()

    def _load_recognizer(self):
        model_name = self.cfg.selected_model

        # FACENET512 (DeepFace internal)
        if model_name == "facenet":
            try:
                from deepface import DeepFace
                print("INFO: DeepFace Facenet512 will be used (represent)")
                return True   # no model object needed
            except Exception as e:
                print(f"ERR: Cannot initialize DeepFace Facenet512: {e}")
                return None

        # YOLO backbone
        if model_name == "yolo_backbone":
            print("INFO: YOLO backbone used for embeddings")
            return None

        # ARC / VGG
        if model_name in ["arcface", "vgg"]:
            try:
                from deepface import DeepFace
                print(f"INFO: DeepFace {model_name} initialized")
                return True
            except Exception as e:
                print(f"ERR: deepface not available: {e}")
                return None

        return None


    def _load_whitelist(self):
        folder = self.cfg.whitelist_dir
        out = []

        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        print(f"INFO: Generating whitelist from `{folder}`...")
        try:
            for f in sorted(os.listdir(folder)):
                if not f.lower().endswith(valid_ext):
                    print(f"SKIP: {f} (not a supported image extension)")
                    continue

                path = os.path.join(folder, f)
                if not os.path.isfile(path):
                    print(f"SKIP: {f} (not a file)")
                    continue

                img = cv2.imread(path)
                if img is None or img.size == 0:
                    print(f"ERR: cannot read image {f}, skipping")
                    continue

                emb = self.get_embedding(img)
                if emb is None:
                    print(f"ERR: no embedding for {f}, skipping")
                    continue

                out.append(emb)
                print(f" - Loaded: {f}")
        except Exception as e:
            print(f"ERR: exception while loading whitelist: {e}")

        print(f"INFO: {len(out)} embeddings loaded.")
        return np.array(out) if len(out) > 0 else np.array([])

    @torch.no_grad()
    def get_embedding(self, img_bgr):
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return None

        model = self.cfg.selected_model

        # --------------------- FACENET ---------------------
        if model == "facenet":
            try:
                from deepface import DeepFace

                res = DeepFace.represent(
                    img_path=img_bgr,
                    model_name="Facenet512",
                    enforce_detection=False,
                    detector_backend="skip"
                )

                if isinstance(res, list) and len(res) > 0:
                    return np.array(res[0]["embedding"], dtype=np.float32)

                return None

            except Exception as e:
                print(f"ERR: Facenet512 embedding failed: {e}")
                return None

        # --------------------- YOLO BACKBONE ---------------------
        if model == "yolo_backbone":
            if self.detector is None:
                return None
            try:
                x = cv2.resize(img_bgr, (128, 128))
                x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

                y = x
                # iterate model backbone safely (some ultralytics versions differ)
                try:
                    layers = getattr(self.detector.model, "model", self.detector.model)
                except Exception:
                    layers = getattr(self.detector, "model", None)

                for i, layer in enumerate(layers):
                    y = layer(y)
                    if i == 9:
                        break

                emb = torch.mean(y, dim=(2, 3)).flatten()
                return (emb / emb.norm()).cpu().numpy()
            except Exception as e:
                print(f"ERR: yolo_backbone embedding failed: {e}")
                return None

        # --------------------- DEEPFACE (ARC/VGG) ---------------------
        if model in ["arcface", "vgg"]:
            try:
                from deepface import DeepFace
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                model_name_map = {"arcface": "ArcFace", "vgg": "VGG-Face"}

                res = DeepFace.represent(
                    img_path=rgb,
                    model_name=model_name_map[model],
                    enforce_detection=False,
                    detector_backend="skip"
                )
                if isinstance(res, list) and len(res) > 0 and "embedding" in res[0]:
                    return np.array(res[0]["embedding"])
                return None
            except Exception as e:
                print(f"ERR: deepface embedding failed: {e}")
                return None

        return None

    def get_distance(self, a, b):
        try:
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except Exception:
            return float("inf")

    def blur(self, img):
        try:
            return cv2.GaussianBlur(img, (51, 51), 30)
        except Exception:
            return img

    def predict_frame(self, frame):
        if frame is None:
            return None, []

        if self.detector is None:
            # if detector missing, return original frame (no detection)
            return frame, []

        sframe = cv2.resize(frame, self.cfg.res)
        proc = sframe.copy()
        detections = []

        # YOLO detect
        try:
            results = self.detector(sframe, stream=True, verbose=False)
        except Exception as e:
            print(f"ERR: YOLO detection failed: {e}")
            return proc, detections

        for r in results:
            try:
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

                    if getattr(self.target_embeddings, "size", 0) > 0:
                        emb = self.get_embedding(face)
                        if emb is not None and len(self.target_embeddings) > 0:
                            dists = [self.get_distance(t, emb) for t in self.target_embeddings]
                            score = min(dists) if len(dists) > 0 else 1.0
                            is_known = score <= self.cfg.threshold

                    # Blur unknown
                    if not is_known:
                        try:
                            proc[y1:y2, x1:x2] = self.blur(proc[y1:y2, x1:x2])
                        except Exception:
                            pass

                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "ok": is_known,
                        "score": float(score)
                    })
            except Exception as e:
                print(f"ERR: exception while parsing detection: {e}")
                continue

        return proc, detections
