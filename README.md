## Crowd Counting and Unique Face Detection

This project focuses on counting people in a crowd and detecting unique faces using four different deep learning models:

- `YOLOv8n`
- `YOLOv8l-face`
- `YOLOv11n`
- `EfficientDet`

Each model is implemented in its own Python file:
- `yolov8n.py`
- `yolov8l-face.py`
- `yolov11n.py`
- `efficientdet.py`

### Installation Requirements

To set up the environment, install all the necessary Python packages:

```bash
pip install ultralytics
pip install face-recognition
pip install deep_sort_realtime
pip install opencv-python
pip install effdet
pip install torchvision
pip install facenet-pytorch
pip install numpy
pip install torch
pip install Pillow scikit-learn
```

### Usage

Make sure to update the video input path in your code:

```python
cap = cv2.VideoCapture('path/to/your/video.mp4')
```

Replace `'path/to/your/video.mp4'` with the actual path to your video file.
