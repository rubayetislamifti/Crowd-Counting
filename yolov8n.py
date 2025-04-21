import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition
import numpy as np

# Load YOLOv8 model
model = YOLO(r'I:\opencv-people-counter\yolov8n.pt')

# Initialize Deep SORT tracker
tracker = DeepSort(
    max_age=60,
    n_init=3,
    max_iou_distance=0.9
)

# Open video file
cap = cv2.VideoCapture(r"I:\opencv-people-counter\kagglehub\datasets\khitthanhnguynphan\crowduit\versions\1\Crowd-UIT\Video\4.mp4")

# Set to store unique person IDs and known face encodings
unique_ids = set()
known_face_encodings = []
total_person_ids = set()
total_face_count = 0

# Frame loop
while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    # Define region of interest
    roi_x_offset = 0
    roi_y_offset = 0
    roi = frame

    # Run YOLOv8 detection
    results = model.predict(roi, stream=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 += roi_x_offset
            x2 += roi_x_offset
            y1 += roi_y_offset
            y2 += roi_y_offset
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        if track.time_since_update == 0:
            # Add to total persons set
            total_person_ids.add(track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            person_frame = frame[y1:y2, x1:x2]

            if person_frame.size == 0:
                continue

            resized_face = cv2.resize(person_frame, (0, 0), fx=2.0, fy=2.0)

            face_locations = face_recognition.face_locations(resized_face)

            if face_locations:
                try:
                    face_encodings = face_recognition.face_encodings(resized_face, face_locations)

                    # 🔢 Count every detected face
                    total_face_count += len(face_encodings)

                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.6)
                        if not any(matches):
                            known_face_encodings.append(encoding)
                            unique_ids.add(track_id)
                            print(f"[NEW FACE] Track ID {track_id} added. Unique: {len(unique_ids)} Total: {total_face_count}")
                except Exception as e:
                    print(f"[!] Face encoding error on track {track_id}: {e}")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw info panel
    # cv2.rectangle(frame, (5, 5), (440, 70), (0, 0, 0), -1)
    cv2.putText(frame, f'Unique People: {len(unique_ids)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    cv2.putText(frame, f'Total Faces Detected: {total_face_count}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    cv2.putText(frame, f'Total People: {len(total_person_ids)}',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    # Show the result
    cv2.imshow('Crowd Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

screenshot_path = "final_screenshot.jpg"
cv2.imwrite(screenshot_path, frame)
print(f"📸 Screenshot saved at: {screenshot_path}")

cap.release()
cv2.destroyAllWindows()

print("Video processing complete.")
print(f"Total unique people detected: {len(unique_ids)}")
print(f"Total face detections in video: {total_face_count}")
print(f"Total people (tracked): {len(total_person_ids)}")

