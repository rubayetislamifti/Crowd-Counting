import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition

# Load YOLOv8 model
model = YOLO(r"C:\Users\bestbrandbd\Downloads\yolov8l-face-lindevs.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=60, n_init=3, max_iou_distance=0.9)

# Open video file
cap = cv2.VideoCapture(r"I:\opencv-people-counter\kagglehub\datasets\khitthanhnguynphan\crowduit\versions\1\Crowd-UIT\Video\10.mp4")

# Get video info for time and progress
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Track unique person IDs and face encodings
unique_person_ids = set()
known_face_encodings = []
all_person_ids = set()

while cap.isOpened():
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    # Run YOLOv8 detection
    results = model.predict(frame, stream=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf > 0.7:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            track_id = track.track_id
            all_person_ids.add(track_id)
            unique_person_ids.add(track_id)

            # Get bounding box and crop face
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # Upscale for better face detection
            upscaled = cv2.resize(person_crop, (0, 0), fx=2.0, fy=2.0)
            face_locations = face_recognition.face_locations(upscaled)

            if face_locations:
                try:
                    face_encodings = face_recognition.face_encodings(upscaled, face_locations)
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.6)
                        if not any(matches):
                            known_face_encodings.append(encoding)
                except Exception as e:
                    print(f"[!] Face encoding error: {e}")

            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Calculate and display time and progress
    remaining_time_seconds = int((total_frames - current_frame) / fps)
    minutes = remaining_time_seconds // 60
    seconds = remaining_time_seconds % 60
    time_text = f'Remaining time: {minutes:02d}:{seconds:02d}'

    percentage = (current_frame / total_frames) * 100
    progress_text = f'Progress: {percentage:.1f}%'

    # Display stats
    cv2.putText(frame, f'Total people tracked: {len(all_person_ids)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    cv2.putText(frame, f'Unique faces detected: {len(known_face_encodings)}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 0, 0), 2)
    cv2.putText(frame, time_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, progress_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show video
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final output
print(f"Total people tracked (YOLO + DeepSORT): {len(all_person_ids)}")
print(f"Total unique faces detected (face recognition): {len(known_face_encodings)}")
