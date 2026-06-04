import sys
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if len(sys.argv) < 3:
    print("0 0 0 0")
    sys.exit(0)

img_path = sys.argv[1]
model_path = sys.argv[2]

img = cv2.imread(img_path)

if img is None:
    print("0 0 0 0")
    sys.exit(0)

h, w = img.shape[:2]

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.4,
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4
)

with PoseLandmarker.create_from_options(options) as landmarker:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print("0 0 0 0")
        sys.exit(0)

    landmarks = result.pose_landmarks[0]

    xs = []
    ys = []

    for lm in landmarks:
        if lm.visibility is None or lm.visibility > 0.3:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if len(xs) == 0:
        print("0 0 0 0")
        sys.exit(0)

    x1 = max(min(xs), 0)
    y1 = max(min(ys), 0)
    x2 = min(max(xs), w - 1)
    y2 = min(max(ys), h - 1)

    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)

    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, w - 1)
    y2 = min(y2 + pad_y, h - 1)

    print(x1, y1, x2 - x1, y2 - y1)
