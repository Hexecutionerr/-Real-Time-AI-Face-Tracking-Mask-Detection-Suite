import cv2
import sys
import config

class CameraStream:
    """
    Manages the webcam hardware stream, optimized for OpenCV and FPS stability.
    Uses Python's Context Manager protocol (with...as) for secure hardware lifecycle management.
    """
    def __init__(self, src=config.CAMERA_INDEX):
        self.src = src
        self.cap = None

    def __enter__(self):
        """Initializes the webcam when entering the 'with' block."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Guarantees hardware release when exiting the 'with' block."""
        self.stop()

    def start(self):
        config.logger.info(f"Connecting to hardware camera (Index {self.src})...")
        self.cap = cv2.VideoCapture(self.src)
        
        if not self.cap.isOpened():
            config.logger.critical(f"Camera index {self.src} inaccessible. Check hardware permissions.")
            sys.exit(1)

        # Apply FPS optimizations via hardware-level downscaling
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            config.logger.info(f"Camera stream locked at {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}.")
        except Exception as e:
            config.logger.warning(f"Failed to enforce resolution: {e}")
            
        return self

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def stop(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            config.logger.info("Camera stream closed.")
