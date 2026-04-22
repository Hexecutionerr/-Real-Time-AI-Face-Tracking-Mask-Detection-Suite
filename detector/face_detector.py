import os
import sys
import cv2
import urllib.request
import numpy as np
import config

try:
    from ultralytics import YOLO
except ImportError:
    config.logger.critical("Missing 'ultralytics'. Run: pip install ultralytics")
    sys.exit(1)

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    config.logger.warning("Missing 'face_recognition'. Identity matching disabled. Run: pip install face_recognition")


class AdvancedFaceDetector:
    """
    Unified OOP class handling YOLOv8 Inference and Deep-Learning Face Recognition.
    """
    def __init__(self):
        self.model = self._load_yolo()
        self.known_encodings = []
        self.known_names = []
        self.recognition_ready = False
        self._init_recognizer()

    def _load_yolo(self):
        """
        Safely loads YOLO weights. Auto-downloads face-specific model if missing.
        """
        if not os.path.exists(config.MODEL_PATH):
            config.logger.info("Downloading YOLOv8 Face weights from HuggingFace...")
            try:
                urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
                config.logger.info("Weights downloaded successfully.")
            except Exception as e:
                config.logger.error(f"Download failed: {e}. Falling back to standard YOLOv8n.")
                return YOLO(config.FALLBACK_MODEL)
        
        config.logger.info("Mounting YOLOv8 model to memory...")
        return YOLO(config.MODEL_PATH)

    def _init_recognizer(self):
        """
        Initializes the face_recognition module by encoding local images.
        """
        if not FACE_REC_AVAILABLE:
            return
            
        if not os.path.exists(config.KNOWN_FACES_DIR):
            os.makedirs(config.KNOWN_FACES_DIR)
            config.logger.info(f"Created {config.KNOWN_FACES_DIR}/ directory for identities.")
            return

        for file in os.listdir(config.KNOWN_FACES_DIR):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(file)[0].replace('_', ' ').title()
                img_path = os.path.join(config.KNOWN_FACES_DIR, file)
                
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(name)
                        config.logger.info(f"Registered identity: {name}")
                except Exception as e:
                    config.logger.warning(f"Error processing {file}: {e}")
                    
        if self.known_encodings:
            self.recognition_ready = True
            config.logger.info(f"Face Recognition active with {len(self.known_names)} subjects.")

    def _recognize_identity(self, frame_rgb, x1, y1, x2, y2):
        """
        Encodes a specific bounding box and matches it against known DB.
        """
        if not self.recognition_ready:
            return "Face", config.COLOR_PRIMARY
            
        h, w = frame_rgb.shape[:2]
        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(w, x2), min(h, y2)
        
        if rx2 <= rx1 or ry2 <= ry1:
            return "Face", config.COLOR_PRIMARY
            
        box = (ry1, rx2, ry2, rx1)
        encodings = face_recognition.face_encodings(frame_rgb, [box])
        
        if not encodings:
            return "Unknown", config.COLOR_UNKNOWN
            
        distances = face_recognition.face_distance(self.known_encodings, encodings[0])
        if len(distances) > 0:
            best_idx = np.argmin(distances)
            if distances[best_idx] <= config.FACE_RECOGNITION_TOLERANCE:
                return self.known_names[best_idx], config.COLOR_KNOWN
                
        return "Unknown", config.COLOR_UNKNOWN

    def _render_ui_overlay(self, frame, x1, y1, x2, y2, label, color):
        """
        Helper method to isolate OpenCV drawing logic from data inference.
        """
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, config.FONT_STYLE, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 10), config.FONT_STYLE, 0.7, config.COLOR_TEXT, 2)

    def process_frame(self, frame):
        """
        Runs full pipeline: YOLO inference, ID recognition, and UI overlay.
        """
        frame = cv2.flip(frame, 1)
        
        # Optimize RGB conversion to occur only if recognition is active
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.recognition_ready else None
        
        # Execute YOLO model
        results = self.model.predict(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        face_count = 0
        
        for result in results:
            for box in result.boxes:
                # Bypass false classes if using standard YOLOv8n
                if int(box.cls[0]) != 0 and result.names[int(box.cls[0])] not in ["face", "Face"]:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Identify Face
                label, color = self._recognize_identity(rgb_frame, x1, y1, x2, y2) if rgb_frame is not None else ("Face Detected", config.COLOR_PRIMARY)
                
                # UI Rendering Delegate
                self._render_ui_overlay(frame, x1, y1, x2, y2, label, color)
                
                face_count += 1
                
        return frame, face_count
