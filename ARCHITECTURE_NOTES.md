# 🧠 Architecture & Codebase Notes

This document serves as an in-depth technical breakdown of the project. It is designed to act as "Study Notes" for understanding how the data flows, what each file does, and the logic behind the algorithms. Ideal for Viva preparation, interviews, and onboarding new developers.

---

## 1. `main.py` (The Entry Point for Identity Recognition)
**What it does:**
This is the central orchestrator for the YOLOv8 pipeline. It does not perform any math or AI predictions itself; it simply connects the Camera to the AI Detector and displays the output.

**How it works:**
* Initializes the `AdvancedFaceDetector` (our AI brain) and `CameraStream` (our hardware eyes).
* Uses a **Context Manager** (`with CameraStream() as camera:`) to ensure the webcam is automatically turned off the moment the script stops, preventing "ghost" webcam processes.
* Captures a frame, tracks the elapsed `time.time()` to calculate **FPS (Frames Per Second)**, and passes the frame to the detector.
* Calls `draw_hud` to print the FPS and Face Count text onto the screen.

---

## 2. `config.py` (The Global Settings)
**What it does:**
Stores all the "Magic Numbers" (colors, thresholds, file paths) in one place.

**How it works:**
* Sets up Python's `logging` module so errors are printed beautifully with timestamps (e.g., `[CRITICAL]`, `[INFO]`) instead of basic `print()` statements.
* Defines `FRAME_WIDTH = 640` and `FRAME_HEIGHT = 480`. By forcing the webcam to capture a smaller resolution, we drastically reduce the amount of pixels the neural network has to process, resulting in higher FPS.
* Holds color tuples like `COLOR_KNOWN = (255, 0, 255)` (Magenta) so UI elements can be changed system-wide by editing just one file.

---

## 3. `utils/camera.py` (Hardware Management)
**What it does:**
An Object-Oriented wrapper around OpenCV's `cv2.VideoCapture()`.

**How it works:**
* Implements `__enter__` and `__exit__` magic methods. This makes the class compatible with the `with` keyword.
* On start, it forces the hardware driver to downscale resolution using `cap.set(cv2.CAP_PROP_FRAME_WIDTH)`.
* On exit, it safely calls `self.cap.release()`.

---

## 4. `detector/face_detector.py` (The YOLOv8 AI Brain)
**What it does:**
This is where the heavy lifting happens. It loads the Deep Learning weights, finds faces, and matches them to known identities.

**How it works:**
* **`_load_yolo()`**: Checks if `yolov8n-face.pt` exists. If not, it automatically downloads it from HuggingFace. It mounts the neural network into RAM using the `ultralytics` package.
* **`_init_recognizer()`**: Scans the `known_faces/` folder. For every `.jpg` it finds, it uses the `face_recognition` library (built on `dlib`) to calculate a 128-dimensional facial embedding (a mathematical representation of the face).
* **`process_frame()`**: 
  1. Runs YOLO on the frame to get bounding boxes `(x1, y1, x2, y2)`.
  2. Crops the detected face out of the frame and passes it to `_recognize_identity()`.
  3. Uses Euclidean Distance to compare the cropped face against all faces stored in `known_faces/`. If the distance is `< 0.6`, it declares a match.
  4. Delegates the drawing of the boxes to a private helper `_render_ui_overlay()` to obey the **Single Responsibility Principle**.

---

## 5. `face_mask_detection.py` (The DNN Mask Algorithm)
**What it does:**
A standalone pipeline that uses a different deep-learning model (SSD ResNet-10) to find faces, and then uses a complex heuristic (math/logic) algorithm to guess if a mask or rumal is covering the face.

**How it works:**
It does not use a pre-trained "Mask/No-Mask" neural network. Instead, it uses 4 logical checks on the face bounding box:
1. **Haar Cascades for Nose/Mouth:** Uses OpenCV's XML files to search the face. If a nose/mouth is *not* found, it adds confidence that the face is covered.
2. **HSV Skin Color Analysis:** Converts the face from BGR to HSV colorspace. It checks if the top half of the face (eyes/forehead) has skin-colored pixels, but the bottom half (mouth/chin) does not.
3. **Color Histogram Comparison:** Compares the overall color distribution of the top half vs. the bottom half using `cv2.compareHist()`. If someone wears a blue mask, the bottom half's color distribution will look entirely different from their forehead.
4. **Weighted Scoring System:** The algorithm combines the above checks into a `confidence` score. If `confidence >= 0.50` (50%), it draws a Green box and declares "MASK". Otherwise, it draws a Red box ("NO MASK").
