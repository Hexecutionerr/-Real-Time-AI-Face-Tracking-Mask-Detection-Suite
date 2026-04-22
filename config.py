import cv2
import logging
import os

# Application-wide Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("FaceTracker")

# Paths and URLs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-face.pt")
MODEL_URL = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
FALLBACK_MODEL = "yolov8n.pt"

# Object Detection Params
CONFIDENCE_THRESHOLD = 0.5
FACE_RECOGNITION_TOLERANCE = 0.6

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# UI/UX Constants
WINDOW_NAME = "High-Performance Face Tracker"
COLOR_PRIMARY = (0, 255, 0)       # Green (Default Face)
COLOR_KNOWN = (255, 0, 255)       # Magenta (Recognized)
COLOR_UNKNOWN = (0, 0, 255)       # Red (Unrecognized)
COLOR_TEXT = (0, 0, 0)            # Black
COLOR_HUD = (0, 255, 255)         # Yellow
FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
