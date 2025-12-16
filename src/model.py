#!/usr/bin/env python3
import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import deepface
from ultralytics import YOLO


# NOTE: MobileFaceNet is still a dickass that lightning is really important

class ActiveModelConfig:
    def __init__(
            self,
            yolo_skip_frames=5,      # YOLO runs every N frames (NEW)
            embedding_skip_frames=30, # Recognition runs every M frames (NEW)
            tracking_distance_threshold=50, # Tracking distance (NEW)
            threshold=0.35,
            res=(640, 480),
            whitelist_dir="../whitelist/",
            ):
        self.yolo_skip_frames = yolo_skip_frames
        self.embedding_skip_frames = embedding_skip_frames
        self.tracking_distance_threshold = tracking_distance_threshold
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
            print(f"ERR:  Failed to load YOLO model: {e}")
            self.detector = None

        # Recognizer (stacking)
        self.recognizer = self._load_recognizer()

        # Whitelist
        self.target_embeddings = self._load_whitelist()

        # Tracking System (NEW)
        self.frame_count = 0
        self.current_faces = []  # List of tracked faces with positions and status

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

        print(f"INFO:  Generating whitelist from `{folder}`...")
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

    def calculate_box_distance(self, box1, box2):
        """Calculate distance between two bounding boxes using center points"""
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2

        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2

        distance = ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5
        return distance

    def match_faces_to_previous(self, new_detections, previous_faces):
        """Match new YOLO detections to previously tracked faces"""
        matched_faces = []
        used_detections = set()

        # Match existing faces to new detections
        for prev_face in previous_faces:
            best_match_idx = -1
            best_distance = float('inf')

            # Find closest new detection
            for i, new_detection in enumerate(new_detections):
                if i in used_detections:
                    continue

                distance = self.calculate_box_distance(prev_face['box'], new_detection)

                if distance < self.cfg.tracking_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = i

            # If good match found, update position but keep status
            if best_match_idx != -1:
                updated_face = prev_face.copy()
                updated_face['box'] = new_detections[best_match_idx]
                matched_faces.append(updated_face)
                used_detections.add(best_match_idx)

        # Add new detections as unknown faces
        for i, new_detection in enumerate(new_detections):
            if i not in used_detections:
                matched_faces.append({
                    'box': new_detection,
                    'ok': False,     # Default unknown
                    'score': 1.0,    # Default score
                    'needs_recognition': True
                })

        return matched_faces

    def predict_frame(self, frame):
        if frame is None:
            return None, []

        if self.detector is None:
            return frame, []

        sframe = cv2.resize(frame, self.cfg.res)
        proc = sframe.copy()

        self.frame_count += 1

        # -------------------------------------------------------------
        # STEP 1: YOLO DETECTION (Independent Timing)
        # -------------------------------------------------------------
        yolo_status = "CACHED"
        if self.frame_count % self.cfg.yolo_skip_frames == 0 or self.frame_count == 1:
            yolo_status = "DETECTING"

            # Run YOLO detection
            new_detections = []
            try:
                results = self.detector(sframe, stream=True, verbose=False)

                for r in results:
                    try:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            h, w = sframe.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)

                            if (x2-x1) > 10 and (y2-y1) > 10:  # Filter too small boxes
                                new_detections.append([x1, y1, x2, y2])
                    except Exception as e:
                        print(f"ERR: exception while parsing detection: {e}")
                        continue
            except Exception as e:
                print(f"ERR: YOLO detection failed: {e}")
                return proc, []

            # Update tracking with new detections
            self.current_faces = self.match_faces_to_previous(new_detections, self.current_faces)

        # -------------------------------------------------------------
        # STEP 2: FACE RECOGNITION (Independent Timing)
        # -------------------------------------------------------------
        embedding_status = "CACHED"
        if self.frame_count % self.cfg.embedding_skip_frames == 0 or self.frame_count == 1:
            embedding_status = "RECOGNIZING"

            # Process recognition for all tracked faces
            for face in self.current_faces:
                x1, y1, x2, y2 = face['box']
                face_img = sframe[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                is_known = False
                score = 1.0

                if getattr(self.target_embeddings, "size", 0) > 0:
                    emb = self.get_embedding(face_img)
                    if emb is not None and len(self.target_embeddings) > 0:
                        dists = [self.get_distance(t, emb) for t in self.target_embeddings]
                        score = min(dists) if len(dists) > 0 else 1.0
                        is_known = score <= self.cfg.threshold

                # Update status
                face['ok'] = is_known
                face['score'] = float(score)
                face['needs_recognition'] = False

        # -------------------------------------------------------------
        # STEP 3: VISUALIZATION (Use tracking data)
        # -------------------------------------------------------------
        detections = []

        for face in self.current_faces:
            x1, y1, x2, y2 = face['box']

            # Blur unknown faces
            if not face['ok']:
                try:
                    proc[y1:y2, x1:x2] = self.blur(proc[y1:y2, x1:x2])
                except Exception:
                    pass

            detections.append({
                "box": [x1, y1, x2, y2],
                "ok": face['ok'],
                "score": face['score']
            })

        return proc, detections

    def get_status_info(self):
        """Get current timing status for UI display"""
        yolo_status = "DETECTING" if self.frame_count % self.cfg.yolo_skip_frames == 0 else "CACHED"
        embedding_status = "RECOGNIZING" if self.frame_count % self.cfg.embedding_skip_frames == 0 else "CACHED"

        return {
            'yolo_status': yolo_status,
            'embedding_status': embedding_status,
            'face_count': len(self.current_faces),
            'frame_count': self.frame_count
        }

    def update_timing(self, yolo_skip=None, embedding_skip=None):
        """Update timing parameters during runtime"""
        if yolo_skip is not None:
            self.cfg.yolo_skip_frames = yolo_skip
            print(f"YOLO timing updated:  every {yolo_skip} frame(s)")

        if embedding_skip is not None:
            self.cfg.embedding_skip_frames = embedding_skip
            print(f"Embedding timing updated: every {embedding_skip} frame(s)")

