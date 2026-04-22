"""
==========================================================================
  FILE        : face_mask_detection.py
  PROJECT     : Real-Time Face Mask Detection System
  DESCRIPTION : Detects faces using a Deep Learning model (SSD + ResNet)
                and classifies whether the person is wearing a mask/face
                covering (rumal, handkerchief, etc.) or not.
  TECHNIQUE   : DNN Face Detection + Haar Cascade Feature Check
                + HSV Skin Color Analysis (Heuristic Approach)
  HOW TO RUN  : python face_mask_detection.py
  QUIT        : Press 'q' or ESC key
==========================================================================
"""

# ========================================================================
# SECTION 1: IMPORT LIBRARIES
# ========================================================================
# cv2       → OpenCV library for computer vision & deep learning inference
# numpy     → Numerical operations on arrays (image = numpy array)
# sys       → System-level operations (exit on error)
# os        → File path handling
# time      → For calculating FPS (Frames Per Second)
# ========================================================================

import cv2
import numpy as np
import sys
import os
import time

# Fix for Windows Terminal Emoji/Unicode Crash
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


# ========================================================================
# SECTION 2: LOAD HAAR CASCADE FILES FOR NOSE & MOUTH DETECTION
# ========================================================================
# What are Haar Cascades for Nose/Mouth?
#   - OpenCV ships with pre-trained cascade XML files for detecting
#     specific facial features like nose, mouth, eyes, etc.
#   - We use mouth & nose cascades to check if these features are
#     VISIBLE or HIDDEN (occluded by a mask/cloth).
#
# Logic:
#   - If mouth and nose are NOT detected → likely covered by mask
#   - If mouth and nose ARE detected → face is uncovered (no mask)
# ========================================================================

def load_feature_cascades():
    """
    Loads Haar Cascade classifiers for nose and mouth detection.
    These cascade files come bundled with the OpenCV installation.

    Returns:
        nose_cascade  → CascadeClassifier for nose (or None if not found)
        mouth_cascade → CascadeClassifier for mouth (or None if not found)
    """
    # Find the directory where OpenCV stores its built-in cascade files
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_dir = os.path.join(cv2_base_dir, 'data')

    # Build full paths to nose and mouth cascade XML files
    nose_path = os.path.join(haar_dir, 'haarcascade_mcs_nose.xml')
    mouth_path = os.path.join(haar_dir, 'haarcascade_mcs_mouth.xml')

    # Attempt to load nose cascade
    nose_cascade = None
    if os.path.exists(nose_path):
        nose_cascade = cv2.CascadeClassifier(nose_path)
        if nose_cascade.empty():
            nose_cascade = None  # Failed to load, set to None

    # Attempt to load mouth cascade
    mouth_cascade = None
    if os.path.exists(mouth_path):
        mouth_cascade = cv2.CascadeClassifier(mouth_path)
        if mouth_cascade.empty():
            mouth_cascade = None  # Failed to load, set to None

    return nose_cascade, mouth_cascade


# ========================================================================
# SECTION 3: MASK DETECTION ALGORITHM (Core Logic)
# ========================================================================
# This is the BRAIN of the project. It uses FOUR techniques combined:
#
#   TECHNIQUE 1 — Haar Cascade Feature Check (Mouth & Nose)
#       → Checks if mouth/nose are VISIBLE in the face region.
#       → If NOT visible = face is likely covered = MASK
#
#   TECHNIQUE 2 — HSV Skin Color Analysis
#       → Converts the face region to HSV color space.
#       → Detects skin-colored pixels in upper vs lower face.
#       → If upper face has skin but lower face does NOT = MASK
#
#   TECHNIQUE 3 — Color Histogram Comparison
#       → Compares the color distribution of upper vs lower face.
#       → If they look very different = cloth/mask is covering lower face.
#       → This is the most ROBUST technique for detecting any type of
#         face covering (surgical mask, rumal, handkerchief, cloth, etc.)
#
#   TECHNIQUE 4 — Weighted Confidence Score
#       → Combines all three techniques into a confidence percentage.
#       → If confidence >= 45% → classified as MASK
#       → If confidence < 45%  → classified as NO MASK
#
# Why HSV (Hue, Saturation, Value)?
#   - HSV separates color information (Hue) from brightness (Value).
#   - This makes skin detection more robust under different lighting.
#   - Skin color in HSV: Hue ≈ 0-35, Saturation ≈ 20-255, Value ≈ 50-255
# ========================================================================

def detect_mask(face_gray, face_color, nose_cascade, mouth_cascade):
    """
    Analyzes a single face region to determine if the person is wearing a mask.

    Args:
        face_gray      → Grayscale cropped face image (for Haar detection)
        face_color     → Color (BGR) cropped face image (for skin analysis)
        nose_cascade   → Haar Cascade for nose detection (can be None)
        mouth_cascade  → Haar Cascade for mouth detection (can be None)

    Returns:
        is_masked   → True if mask detected, False otherwise
        confidence  → Float between 0.0 and 1.0 (how confident we are)
    """
    h, w = face_gray.shape[:2]  # Get height and width of the face region

    # Skip very small face regions (too small to analyze reliably)
    if h < 20 or w < 20:
        return False, 0.0

    # Apply histogram equalization to improve contrast for Haar detection
    # This helps the cascade classifiers work better under varied lighting
    face_gray_eq = cv2.equalizeHist(face_gray)

    # ----------------------------------------------------------------
    # TECHNIQUE 1: MOUTH & NOSE OCCLUSION CHECK
    # ----------------------------------------------------------------
    # We look for nose in the middle region and mouth in the bottom region.
    # If they are NOT found → the face is likely covered (mask/rumal).
    # ----------------------------------------------------------------

    nose_detected = False
    mouth_detected = False

    # --- Nose Detection ---
    if nose_cascade is not None:
        # Extract the MIDDLE region of the face (where nose typically is)
        # Rows: from 1/3 height to 80% height
        # Columns: from 1/4 width to 3/4 width (center strip)
        mid_region = face_gray_eq[h // 3 : int(h * 0.8), w // 4 : 3 * w // 4]

        noses = nose_cascade.detectMultiScale(
            mid_region,
            scaleFactor=1.1,                    # Image pyramid reduction
            minNeighbors=3,                     # Very aggressive to ensure we detect visible noses
            minSize=(int(w * 0.08), int(h * 0.08))  # Minimum nose size
        )
        nose_detected = len(noses) > 0  # True if at least one nose found

    # --- Mouth Detection ---
    if mouth_cascade is not None:
        # Extract the BOTTOM HALF of the face (where mouth typically is)
        mouth_region = face_gray_eq[h // 2 :, :]

        mouths = mouth_cascade.detectMultiScale(
            mouth_region,
            scaleFactor=1.1,
            minNeighbors=8,                        # Higher to avoid false detections on cloth/rumal
            minSize=(int(w * 0.12), int(h * 0.04))  # Minimum mouth size
        )
        mouth_detected = len(mouths) > 0  # True if at least one mouth found

    # ----------------------------------------------------------------
    # TECHNIQUE 2: SKIN COLOR ANALYSIS USING HSV
    # ----------------------------------------------------------------
    # We compare skin visibility between UPPER face and LOWER face.
    # Upper face (forehead, eyes) → always visible even with mask
    # Lower face (nose, mouth, chin) → covered if mask is worn
    #
    # If upper face shows skin but lower face doesn't → MASK!
    # ----------------------------------------------------------------

    # Convert the face region from BGR to HSV color space
    hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)

    # Define TWO HSV ranges for skin color detection to cover diverse skin tones
    # Range 1: Lighter skin tones (Hue 0-25)
    lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
    # Range 2: Darker/olive skin tones (Hue 25-40)
    lower_skin2 = np.array([25, 20, 50], dtype=np.uint8)
    upper_skin2 = np.array([40, 255, 255], dtype=np.uint8)

    # Create binary masks for both ranges and combine them
    skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)

    # Apply morphological opening to remove noise from the skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Split the skin mask into UPPER half and LOWER half
    upper_half = skin_mask[: h // 2, :]   # Top half of face
    lower_half = skin_mask[h // 2 :, :]   # Bottom half of face

    # Calculate skin pixel density (percentage of skin pixels)
    upper_skin_density = np.sum(upper_half > 0) / (upper_half.size + 1e-6)
    lower_skin_density = np.sum(lower_half > 0) / (lower_half.size + 1e-6)

    # ----------------------------------------------------------------
    # TECHNIQUE 3: COLOR HISTOGRAM COMPARISON
    # ----------------------------------------------------------------
    # Compare the color distribution between UPPER and LOWER face.
    # If someone wears a mask/rumal, the lower half will have a very
    # DIFFERENT color pattern than the upper half (skin vs cloth).
    # We use HSV histograms and cv2.compareHist() for this comparison.
    #
    # Correlation value ranges from -1 to 1:
    #   1.0  = identical color distribution (no mask)
    #   0.0  = no correlation (very different = mask)
    #  -1.0  = opposite distribution (definitely different)
    # ----------------------------------------------------------------

    upper_hsv = hsv[: h // 2, :]   # Upper face HSV
    lower_hsv = hsv[h // 2 :, :]   # Lower face HSV

    # Calculate 2D histograms (Hue + Saturation) for both halves
    hist_upper = cv2.calcHist([upper_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_upper, hist_upper)  # Normalize for fair comparison

    hist_lower = cv2.calcHist([lower_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_lower, hist_lower)  # Normalize for fair comparison

    # Compare histograms using correlation method
    hist_similarity = cv2.compareHist(hist_upper, hist_lower, cv2.HISTCMP_CORREL)

    # ----------------------------------------------------------------
    # TECHNIQUE 4: WEIGHTED CONFIDENCE SCORING
    # ----------------------------------------------------------------
    # We combine all techniques into a single confidence score.
    # The user priority is NOSE. If nose is visible, NOT MASK. If covered, MASK.
    #
    # Weights:
    #   Nose occlusion             → 60% (primary decisive check)
    #   Color histogram comparison → 20% (robustness for covering)
    #   Skin color analysis        → 10% (skin vs non-skin)
    #   Mouth occlusion            → 10% (supplementary)
    #
    # Final decision: confidence >= 0.50 → MASK
    # ----------------------------------------------------------------

    mask_score = 0.0      # Accumulates evidence FOR mask
    total_weight = 0.0    # Accumulates total possible weight

    # --- Nose Occlusion Score (60% weight) ---
    if nose_cascade is not None:
        total_weight += 0.60
        if not nose_detected:
            # Nose NOT visible → PRIMARY evidence of mask
            mask_score += 0.60

    # --- Color Histogram Score (20% weight) ---
    total_weight += 0.20
    if hist_similarity < 0.4:
        # Upper and lower face look VERY different → strong mask evidence
        mask_score += 0.20
    elif hist_similarity < 0.6:
        # Moderately different → partial evidence
        mask_score += 0.10
    elif hist_similarity < 0.75:
        # Slightly different → weak evidence
        mask_score += 0.05

    # --- Skin Color Score (10% weight) ---
    if upper_skin_density > 0.05:
        # Person has visible skin on the upper face (forehead/eyes area)
        total_weight += 0.10

        skin_ratio = lower_skin_density / (upper_skin_density + 1e-6)

        if skin_ratio < 0.35:
            # Lower face has MUCH LESS skin than upper → strongly suggests mask
            mask_score += 0.10
        elif skin_ratio < 0.60:
            # Lower face has somewhat less skin → partial evidence
            mask_score += 0.05

    # --- Mouth Occlusion Score (10% weight) ---
    if mouth_cascade is not None:
        total_weight += 0.10
        if not mouth_detected:
            # Mouth NOT visible → strong evidence of mask
            mask_score += 0.10

    # --- Calculate Final Confidence ---
    if total_weight > 0:
        confidence = mask_score / total_weight  # Normalize to 0.0 - 1.0
    else:
        confidence = 0.0  # Fallback if no technique was applicable

    # --- Final Decision ---
    # Threshold: 50% confidence to classify as masked
    is_masked = confidence >= 0.50  # Threshold: 50% confidence

    return is_masked, confidence


# ========================================================================
# SECTION 4: LOAD DNN (Deep Neural Network) FACE DETECTOR
# ========================================================================
# What is DNN Face Detection?
#   - Uses a pre-trained deep learning model called SSD (Single Shot
#     Detector) with a ResNet-10 backbone.
#   - Much MORE ACCURATE than Haar Cascade, especially for:
#       → Partially covered faces (masks, sunglasses)
#       → Faces at different angles (tilted, side view)
#       → Faces in poor lighting conditions
#
# Model Files:
#   - deploy.prototxt → Defines the network architecture (layers, connections)
#   - res10_300x300_ssd_iter_140000.caffemodel → Pre-trained weights (10MB)
#
# How it works:
#   1. Input image is resized to 300x300 pixels
#   2. Image is converted to a "blob" (preprocessed tensor)
#   3. Blob is passed through the neural network
#   4. Network outputs a list of detected face regions with confidence scores
# ========================================================================

def load_dnn_face_detector():
    """
    Loads the SSD + ResNet DNN model for face detection.
    Returns the loaded neural network object.
    Exits if model files are not found.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to the DNN model files
    prototxt_path = os.path.join(base_dir, "deploy.prototxt")
    weights_path = os.path.join(base_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    # Verify both files exist
    if not os.path.exists(prototxt_path):
        print(f"[ERROR] Model architecture file not found: deploy.prototxt")
        sys.exit(1)

    if not os.path.exists(weights_path):
        print(f"[ERROR] Model weights file not found: res10_300x300_ssd_iter_140000.caffemodel")
        sys.exit(1)

    # Load the Caffe model into OpenCV's DNN module
    # readNetFromCaffe() reads models trained in the Caffe deep learning framework
    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

    return net


# ========================================================================
# SECTION 5: DNN FACE DETECTION IN A SINGLE FRAME
# ========================================================================
# Steps:
#   1. Resize the frame to 300x300 (model's expected input size)
#   2. Create a "blob" — a preprocessed image tensor with:
#       - Scale factor 1.0 (no scaling)
#       - Mean subtraction (104, 177, 123) for normalization
#   3. Feed the blob through the network
#   4. Extract face bounding boxes with confidence > 50%
# ========================================================================

def detect_faces_dnn(frame, net):
    """
    Uses the DNN model to detect all faces in a frame.

    Args:
        frame → The input BGR image (numpy array)
        net   → The loaded DNN model

    Returns:
        face_boxes → List of (startX, startY, endX, endY) tuples
    """
    (h, w) = frame.shape[:2]  # Get frame dimensions

    # Create a blob from the image
    # blobFromImage() preprocesses the image for the neural network:
    #   - Resizes to 300x300
    #   - Applies mean subtraction (104.0, 177.0, 123.0) to normalize pixel values
    #   - This mean was used during model training, so we must use the same
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),  # Resize to model's input dimensions
        1.0,                             # Scale factor (no scaling)
        (300, 300),                      # Spatial size for output image
        (104.0, 177.0, 123.0)            # Mean subtraction values (BGR)
    )

    # Pass the blob through the network to get detections
    net.setInput(blob)        # Set the blob as input
    detections = net.forward()  # Run forward pass → get output predictions

    # Parse detections and filter by confidence
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence score (0 to 1)

        if confidence > 0.3:  # Lowered to 30% to detect partially covered faces (mask/rumal)
            # Extract bounding box coordinates
            # The model outputs coordinates as fractions of 300x300
            # We multiply by actual frame size to get real pixel coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp coordinates to stay within frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Skip tiny detections (likely noise)
            if (endX - startX) < 20 or (endY - startY) < 20:
                continue

            face_boxes.append((startX, startY, endX, endY))

    return face_boxes


# ========================================================================
# SECTION 6: DRAW RESULTS ON THE FRAME
# ========================================================================
# For each detected face, we:
#   1. Classify it as MASK or NO MASK using detect_mask()
#   2. Draw a colored bounding box:
#       - GREEN for MASK
#       - RED for NO MASK
#   3. Display the label with confidence percentage
#   4. Draw a guidance line at the 1/3 mark (upper/lower face boundary)
# ========================================================================

def annotate_frame(frame, face_boxes, gray, nose_cascade, mouth_cascade):
    """
    Draws bounding boxes, labels, and mask/no-mask classification
    on each detected face.

    Args:
        frame          → The original BGR frame to draw on
        face_boxes     → List of face bounding box coordinates
        gray           → Grayscale version of the frame
        nose_cascade   → Haar Cascade for nose detection
        mouth_cascade  → Haar Cascade for mouth detection

    Returns:
        frame          → Annotated frame
        mask_count     → Number of faces with masks
        no_mask_count  → Number of faces without masks
    """
    # Define colors (BGR format)
    COLOR_MASK = (0, 255, 0)       # Green → Mask detected
    COLOR_NO_MASK = (0, 0, 255)    # Red → No mask detected

    mask_count = 0
    no_mask_count = 0

    for (startX, startY, endX, endY) in face_boxes:
        # Extract face region (ROI = Region of Interest) for analysis
        face_gray = gray[startY:endY, startX:endX]
        face_color = frame[startY:endY, startX:endX]

        # Classify this face as MASK or NO MASK
        is_masked, confidence = detect_mask(
            face_gray, face_color,
            nose_cascade, mouth_cascade
        )

        # Set label, color, and counter based on classification result
        if is_masked:
            mask_count += 1
            color = COLOR_MASK
            label = "MASK"
            display_conf = f"{confidence * 100:.0f}%"
        else:
            no_mask_count += 1
            color = COLOR_NO_MASK
            label = "NO MASK"
            display_conf = f"{(1 - confidence) * 100:.0f}%"

        # Draw bounding box around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Draw label background and text above the bounding box
        label_text = f"{label} {display_conf}"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame,
            (startX, startY - text_h - 15),
            (startX + text_w + 10, startY),
            color, -1  # Filled rectangle for label background
        )
        cv2.putText(
            frame, label_text,
            (startX + 5, startY - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2  # White text on colored background
        )

        # Draw a horizontal guide line at 1/3 mark
        # This visually shows the upper/lower face boundary used in analysis
        face_h = endY - startY
        mid_y = startY + face_h // 3
        cv2.line(frame, (startX, mid_y), (endX, mid_y), (255, 255, 0), 1)

    return frame, mask_count, no_mask_count


# ========================================================================
# SECTION 7: DRAW HUD (Heads-Up Display) — INFO PANEL
# ========================================================================
# A semi-transparent dark panel at the top of the screen showing:
#   - Project title
#   - FPS counter (frames per second)
#   - Count of masked faces
#   - Count of unmasked faces
#   - Total face count
#
# We use cv2.addWeighted() to create a transparency/overlay effect.
# ========================================================================

def draw_hud(frame, fps, mask_count, no_mask_count):
    """
    Draws a semi-transparent info panel at the top of the frame.
    """
    # Define colors
    COLOR_MASK = (0, 255, 0)       # Green
    COLOR_NO_MASK = (0, 0, 255)    # Red
    COLOR_INFO = (0, 255, 255)     # Yellow

    panel_height = 80

    # Create a copy of the frame and draw a solid black rectangle on it
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)

    # Blend the overlay with the original frame (60% overlay, 40% original)
    # This creates a semi-transparent dark panel effect
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text information on the panel
    cv2.putText(frame, "Face Mask Detection (DNN)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 2)

    cv2.putText(frame, f"With Mask: {mask_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MASK, 2)

    cv2.putText(frame, f"Without Mask: {no_mask_count}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NO_MASK, 2)

    total = mask_count + no_mask_count
    cv2.putText(frame, f"Total Faces: {total}", (frame.shape[1] - 200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


# ========================================================================
# SECTION 8: MAIN FUNCTION — TIES EVERYTHING TOGETHER
# ========================================================================
# Execution Flow:
#   1. Load the DNN face detector (SSD + ResNet model)
#   2. Load Haar Cascades for nose & mouth feature detection
#   3. Open the webcam and set resolution
#   4. Enter the main loop:
#       a. Capture a frame from webcam
#       b. Flip the frame horizontally (mirror effect)
#       c. Convert to grayscale (needed for Haar Cascade)
#       d. Detect faces using DNN model
#       e. For each face → classify as MASK or NO MASK
#       f. Draw annotations and HUD on the frame
#       g. Display the frame in a window
#       h. Calculate and update FPS
#       i. Check for user quit ('q' or ESC)
#   5. On exit: release webcam and close windows
# ========================================================================

def main():
    """
    Main entry point for the Face Mask Detection system.
    Orchestrates loading models, capturing video, and running detection.
    """

    # ---- Step 1: Load the DNN Face Detector ----
    print("=" * 55)
    print("  Loading DNN Face Detector (SSD + ResNet)...")
    net = load_dnn_face_detector()
    print("  ✅ DNN model loaded successfully!")

    # ---- Step 2: Load Haar Cascades for nose & mouth ----
    print("  Loading Haar Cascades for feature detection...")
    nose_cascade, mouth_cascade = load_feature_cascades()
    print("  ✅ Feature detectors ready!")
    print("=" * 55)

    # ---- Step 3: Open the webcam ----
    cap = cv2.VideoCapture(0)  # 0 = default camera

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera connection.")
        sys.exit(1)

    # Set webcam resolution to 640x480 for balanced quality and speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n  🎥 Real-Time Face Mask Detection Started (DNN Mode)")
    print("  Press 'q' or ESC to quit")
    print("=" * 55)

    # ---- FPS Tracking Variables ----
    fps_counter = 0       # Counts frames in the current second
    fps = 0               # Displayed FPS value
    fps_time = time.time()  # Timestamp of last FPS update

    # ---- Mask Alert State ----
    mask_notified = False

    # ---- Step 4: Main Detection Loop ----
    while True:
        # Capture one frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to capture frame.")
            break

        # Flip frame horizontally for a natural mirror-like view
        frame = cv2.flip(frame, 1)

        # Convert to grayscale (needed for Haar Cascade processing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Detect faces using DNN ----
        face_boxes = detect_faces_dnn(frame, net)

        # ---- Classify each face and draw annotations ----
        frame, mask_count, no_mask_count = annotate_frame(
            frame, face_boxes, gray,
            nose_cascade, mouth_cascade
        )

        # ---- Print Alert Once ----
        if mask_count > 0 and not mask_notified:
            print("\n  😷 ALERT: Mask detect hogya!")
            mask_notified = True

        # ---- Calculate FPS ----
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_time >= 1.0:
            fps = fps_counter          # FPS = frames counted in last 1 second
            fps_counter = 0            # Reset counter
            fps_time = current_time    # Reset timer

        # ---- Draw the HUD (info panel) ----
        frame = draw_hud(frame, fps, mask_count, no_mask_count)

        # ---- Display the result ----
        cv2.imshow("Face Mask Detection - Press 'q' to quit", frame)

        # ---- Handle key press ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    # ---- Step 5: Cleanup ----
    cap.release()               # Release the webcam
    cv2.destroyAllWindows()     # Close all OpenCV windows
    print("\n  ✅ Face Mask Detection stopped. Resources released.")


# ========================================================================
# SECTION 9: SCRIPT ENTRY POINT
# ========================================================================
# This ensures main() only runs when this script is executed directly,
# not when imported as a module by another Python file.
# ========================================================================

if __name__ == "__main__":
    main()
