#!/usr/bin/env python3
import cv2
import numpy as np
from deepface import DeepFace
import os

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
frontal_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

print("INFO: Reading the `whitelist` directory...")
directory_path = 'whitelist/'

filename = []
try:
    for entry_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry_name)
        if os.path.isfile(full_path):
            filename.append(full_path)
except Exception as e:
    print(f"An error occurred: {e}")
    filename = []

target_embeddings = []
print("INFO: Embbeding the whitelisted face(s)...")
if not filename:
    print("WARNING: No image files found to embed.")
else:
    print(f"INFO: Embedding {len(filename)} whitelisted face(s)...")
    for img_path in filename:
        try:
            embedding_result = DeepFace.represent(
                    img_path=img_path,
                    model_name="VGG-Face",
                    # detector_backend="retinaface",
                    enforce_detection=False
                    )

            if embedding_result and "embedding" in embedding_result[0]:
                target_embeddings.append(embedding_result[0]["embedding"])
            else:
                print(f"WARNING: Could not find face/embedding for {img_path}. Skipping.")

        except Exception as e:
            print(f"ERROR: Failed to embed {img_path}. Error: {e}")

print(f"INFO: Successfully generated {len(target_embeddings)} embeddings.")

cap = cv2.VideoCapture(0)
target_dims = (320, 240)
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame (stream end?). Exiting...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sframe = cv2.resize(frame, target_dims, interpolation=cv2.INTER_LINEAR)

    try:
        gray = cv2.cvtColor(sframe, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces_frontal = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces_profile = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = list(faces_frontal) + list(faces_profile)

        for (x, y, w, h) in faces:
            face_img = sframe[y:y+h, x:x+w]

        # Use DeepFace to extract all faces using the RetinaFace detector
        # detected_faces = DeepFace.extract_faces(
        #     img_path=sframe,
        #     detector_backend="retinaface",
        #     enforce_detection=True
        # )
        # # Iterate over the detected faces
        # # Each face in 'detected_faces' contains 'facial_area' (x, y, w, h)
        # for face_info in detected_faces:
        #     # Extract the bounding box coordinates (x, y, w, h)
        #     x = face_info['facial_area']['x']
        #     y = face_info['facial_area']['y']
        #     w = face_info['facial_area']['w']
        #     h = face_info['facial_area']['h']
        #
        #     face_img = sframe[y:y+h, x:x+w]

            if not target_embeddings:
                is_whitelisted = False
            else:
                # a. Identifikasi (Recognition)
                # Convert the cropped face image to its embedding
                current_embedding_list = DeepFace.represent(
                    img_path=face_img,
                    model_name="VGG-Face",
                    # detector_backend="retinaface",
                    enforce_detection=False
                )

                # Ensure embedding was successfully created
                if not current_embedding_list:
                    print("WARNING: Failed to generate embedding for detected face. Skipping.")
                    continue

                current_embedding = np.array(current_embedding_list[0]["embedding"])

                # Initialize tracking variables
                min_distance = float('inf')
                is_whitelisted = False
                threshold = 0.40 # VGG-Face threshold for Cosine distance

                # b. Hitung Kemiripan (Cosine Similarity)
                # --- START OF FOR LOOP IMPLEMENTATION ---

                # Loop through every pre-calculated whitelisted embedding
                for target_embed_list in target_embeddings:
                    # Convert target list to NumPy array for calculation
                    target_embedding = np.array(target_embed_list)

                    # Cosine Distance Calculation Formula: 1 - Cosine Similarity
                    # Cosine Similarity = (A . B) / (||A|| * ||B||)
                    a = np.matmul(target_embedding, current_embedding)
                    b = np.linalg.norm(target_embedding)
                    c = np.linalg.norm(current_embedding)

                    distance = 1 - (a / (b * c))

                    # Update the minimum distance found so far
                    if distance < min_distance:
                        min_distance = distance
                # --- END OF FOR LOOP IMPLEMENTATION ---

                # c. Logika Keputusan
                if min_distance <= threshold:
                    is_whitelisted = True

            # d. Terapkan Aksi
            if not is_whitelisted:
                sframe[y:y+h, x:x+w] = cv2.GaussianBlur(face_img, (99, 99), 30)

        cv2.imshow('LIVE CAMERA FEED', sframe)
    except Exception as e:
        print(f"ERROR: Loop stopped with error: {e}")
cap.release()
cv2.destroyAllWindows()
