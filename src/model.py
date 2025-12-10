#!/usr/bin/env python3
import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import deepface
from ultralytics import YOLO

class ActiveModelConfig:
    def __init__(
            self,
            skip_frames=20,
            threshold=0.47,
            res=(640, 480),
            whitelist_dir="../whitelist/",
            ):
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
        from mfnet import MobileFacenet
        model = MobileFacenet()

        ckpt = torch.load("./model/068.ckpt", map_location=self.device)
        model.load_state_dict(ckpt["net_state_dict"], strict=False)
        model.eval()
        model.to(self.device)

        print("INFO: MobileFaceNet loaded")
        return model

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

                # Try YOLO-based crop first (for consistency with live detection)
                face_img = None
                if self.detector is not None:
                    try:
                        res = self.detector(img, verbose=False)
                        for r in res:
                            if len(r.boxes) > 0:
                                x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
                                h, w = img.shape[:2]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                face_img = img[y1:y2, x1:x2]
                                break
                    except Exception as e:
                        print(f"ERR: YOLO crop failed for {f}: {e}")

                if face_img is None or face_img.size == 0:
                    face_img = img  # fallback to full image

                emb = self.get_embedding(face_img)
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
        if img_bgr is None or img_bgr.size == 0:
            return None

        # Convert to RGB
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to 96 × 112 (width, height)
        img = cv2.resize(img, (96, 112))

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        emb = self.recognizer(img)[0].cpu().numpy()

        # Normalize embedding (L2)
        emb = emb / np.linalg.norm(emb)

        return emb

    def get_distance(self, a, b):
        return 1 - np.dot(a, b)

    def blur(self, img):
        try:
            return cv2.GaussianBlur(img, (51, 51), 30)
        except Exception:
            return img

    def predict_frame(self, frame):
        if frame is None:
            return None, []

        if self.detector is None:
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
