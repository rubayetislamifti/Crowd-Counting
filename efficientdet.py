import torch
from effdet import create_model, get_efficientdet_config
from effdet.data import resolve_input_config
from effdet import DetBenchPredict
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity

# ---- Device Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- EfficientDet Setup ----
model_name = 'tf_efficientdet_d0'
model = create_model(model_name, pretrained=True)
model = DetBenchPredict(model).to(device)
model.eval()

config = get_efficientdet_config(model_name)
input_config = resolve_input_config({}, model_config=config)
input_size = input_config['input_size'][1:]  # (width, height)

# ---- Letterbox Resize ----
def letterbox_image(image, expected_size):
    iw, ih = image.size
    ew, eh = expected_size
    scale = min(ew / iw, eh / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", (ew, eh), (128, 128, 128))
    paste_position = ((ew - nw) // 2, (eh - nh) // 2)
    new_image.paste(image_resized, paste_position)
    return new_image, scale, paste_position, (nw, nh)

# ---- Deep SORT ----
deepsort = DeepSort(max_age=30)

# ---- FaceNet Setup ----
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---- Load Video ----
video_path = r"I:\kagglehub\datasets\khitthanhnguynphan\crowduit\versions\1\Crowd-UIT\Video\9.mp4"
cap = cv2.VideoCapture(video_path)

# ---- Video Writer Setup ----
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = 'annotated_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ---- Init Count State ----
frame_count = 0
total_people = 0
face_embeddings_list = []

# ---- ROI Setup ----
roi_x_offset = 0
roi_y_offset = 0

# ---- Main Loop ----
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count > 200:
        break

    roi = frame[roi_y_offset:height, roi_x_offset:width]

    image_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    padded_img, scale, pad, resized_shape = letterbox_image(image_pil, input_size)
    nw, nh = resized_shape

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    ])
    input_tensor = transform(padded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()

    if outputs.shape[0] == 0:
        out.write(frame)
        frame_count += 1
        continue

    boxes, scores, labels = outputs[:, :4], outputs[:, 4], outputs[:, 5]
    person_boxes = [box for box, score, label in zip(boxes, scores, labels)
                    if score > 0.4 and int(label) == 1]
    person_scores = [score for score, label in zip(scores, labels)
                     if score > 0.4 and int(label) == 1]

    xywhs, confs = [], []
    for box, score in zip(person_boxes, person_scores):
        x1, y1, x2, y2 = box
        pad_x, pad_y = pad

        x1 = ((x1 - pad_x) / scale)
        y1 = ((y1 - pad_y) / scale)
        x2 = ((x2 - pad_x) / scale)
        y2 = ((y2 - pad_y) / scale)

        x1 = np.clip(x1, 0, width)
        x2 = np.clip(x2, 0, width)
        y1 = np.clip(y1, 0, height)
        y2 = np.clip(y2, 0, height)

        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2, y1 + h / 2
        xywhs.append([x, y, w, h])
        confs.append(score)

    if xywhs:
        detections = [([x, y, w, h], conf) for ([x, y, w, h], conf) in zip(xywhs, confs)]
        tracks = deepsort.update_tracks(detections, frame=roi)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            pad_size = 20
            x1, y1 = max(x1 - pad_size, 0), max(y1 - pad_size, 0)
            x2, y2 = min(x2 + pad_size, roi.shape[1]), min(y2 + pad_size, roi.shape[0])

            face_crop = roi[y1:y2, x1:x2]
            color = (0, 0, 255)
            label = f"ID: {track_id}"

            if face_crop.size != 0:
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_crop_rgb)

                try:
                    face_img = mtcnn(face_pil)
                    if face_img is not None:
                        embedding = facenet(face_img.unsqueeze(0).to(device)).detach().cpu().numpy()
                        matched = False
                        for known_embed in face_embeddings_list:
                            sim = cosine_similarity(embedding, known_embed)[0][0]
                            if sim > 0.9:
                                matched = True
                                break

                        if not matched:
                            face_embeddings_list.append(embedding)
                            total_people += 1
                            color = (0, 255, 0)
                except Exception as e:
                    print(f"Face detection failed: {e}")

            cv2.rectangle(roi, (x1, y1), (x2, y2), color, 2)
            cv2.putText(roi, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"Total People: {total_people}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Unique Faces: {len(face_embeddings_list)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    out.write(frame)
    frame_resized = cv2.resize(frame, (640, 360))
    cv2.imshow('Annotated Video', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    print(f"Frame {frame_count}, Total People Count: {total_people}")
    print(f"Unique Face Count: {len(face_embeddings_list)}")

# ---- Cleanup ----
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Final Count: {total_people}")
