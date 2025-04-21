from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition

# ------------------ PART 1: Train and Export YOLO Model ------------------ #

# Load a pretrained YOLO11n model
model_train = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 50 epochs
train_results = model_train.train(
    data="coco8.yaml",
    epochs=10,
    imgsz=640,
    device="cpu",
)

# Evaluate the model
metrics = model_train.val()

# Export the trained model to ONNX format
export_path = model_train.export(format="onnx")
print(f"✅ Model exported to: {export_path}")

# ------------------ PART 2: Detection with Total and Unique Face Count ------------------ #

# Load YOLO model (use trained or pretrained)
model_detect = YOLO("yolo11n.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=60, n_init=3, max_iou_distance=0.9)

# Open video file
cap = cv2.VideoCapture(r"I:\kagglehub\datasets\khitthanhnguynphan\crowduit\versions\1\Crowd-UIT\Video\2.mp4")

# Sets to store track IDs and face encodings
unique_person_ids = set()
unique_face_encodings = []

max_people_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    # Run YOLO detection
    results = model_detect(frame, stream=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf > 0.5:  # Person class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Track people
    tracks = tracker.update_tracks(detections, frame=frame)

    frame_people_count = 0

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        unique_person_ids.add(track_id)
        frame_people_count += 1

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Face detection inside person bounding box
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        upscaled = cv2.resize(person_crop, (0, 0), fx=2.0, fy=2.0)
        face_locations = face_recognition.face_locations(upscaled)

        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(upscaled, face_locations)
                for encoding in face_encodings:
                    matches = face_recognition.compare_faces(unique_face_encodings, encoding, tolerance=0.6)
                    if not any(matches):
                        unique_face_encodings.append(encoding)
            except Exception as e:
                print(f"[!] Face encoding error: {e}")

    max_people_count = max(max_people_count, frame_people_count)

    # Display stats on frame
    cv2.putText(frame, f'Current People: {frame_people_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Total People: {len(unique_person_ids)}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Unique Faces: {len(unique_face_encodings)}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Crowd Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final summary
print(f"Video ended.")
print(f"Total unique people tracked: {len(unique_person_ids)}")
print(f"Total unique faces detected: {len(unique_face_encodings)}")
