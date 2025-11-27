import cv2
from ultralytics import YOLO

def main():
    # 1. Load the YOLOv11 model
    # 'yolo11n.pt' is the Nano version (fastest, best for CPU/webcam).
    # It will automatically download the weight file on the first run.
    print("Loading YOLOv11 model...")
    # model = YOLO("yolo11n.pt")
    model = YOLO("model.pt")

    # 2. Initialize the webcam
    # Use 0 for the default camera. If you have external cams, try 1 or 2.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam. Press 'q' to exit.")

    while True:
        # 3. Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame.")
            break

        # 4. Run YOLOv11 inference on the frame
        # stream=True is more memory efficient for video loops
        # verbose=False keeps the terminal output clean
        results = model(frame, stream=True, verbose=False)

        # 5. Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the class ID (0 is 'person' in standard COCO models)
                class_id = int(box.cls[0])

                # Filter: Only draw box if detected class is a Person (ID 0)
                # If you use a custom face-trained model, you might not need this check.
                if class_id == 0:
                    # Get coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get confidence score
                    confidence = float(box.conf[0])

                    # Draw the bounding box (Color: Green, Thickness: 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label text "Person" or "Face" depending on your model
                    label = f"Person: {confidence:.2f}"

                    # Calculate text position (just above the box)
                    t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3

                    # Draw filled rectangle for text background for better visibility
                    cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)

                    # Put the text
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # 6. Display the frame
        cv2.imshow("YOLOv11 Detection", frame)

        # 7. Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
