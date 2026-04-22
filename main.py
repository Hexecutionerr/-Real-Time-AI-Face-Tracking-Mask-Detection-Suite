import cv2
import time
import sys
import config
from utils.camera import CameraStream
from detector.face_detector import AdvancedFaceDetector

def draw_hud(frame, face_count, fps):
    """
    Renders diagnostic telemetry directly onto the video feed.
    """
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), config.FONT_STYLE, 1, config.COLOR_HUD, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), config.FONT_STYLE, 1, (255, 0, 0), 2)
    return frame

def main():
    """
    Primary execution loop containing graceful error handling and system lifecycle logic.
    """
    config.logger.info("=" * 50)
    config.logger.info("Initializing Advanced AI Face Tracker")
    config.logger.info("=" * 50)

    # Initialize Core Architecture Components
    detector = AdvancedFaceDetector()
    
    # Secure Context Manager for Hardware Resource Lifecycle
    with CameraStream() as camera:
        prev_time = time.time()
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    config.logger.error("Video stream lost.")
                    break

                # Calculate precise FPS processing latency
                curr_time = time.time()
                elapsed = curr_time - prev_time
                fps = 1 / elapsed if elapsed > 0 else 0
                prev_time = curr_time

                # Inference & Drawing Pipeline
                frame, count = detector.process_frame(frame)
                frame = draw_hud(frame, count, fps)

                # Display feed
                cv2.imshow(config.WINDOW_NAME, frame)

                # Polling listener (Esc or Q to exit)
                if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
                    config.logger.info("Exit command received.")
                    break

        except Exception as e:
            config.logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            # Context manager handles camera.stop(), just clean up OpenCV UI
            cv2.destroyAllWindows()
            config.logger.info("System shut down cleanly.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        config.logger.info("Process interrupted (Ctrl+C).")
        sys.exit(0)
