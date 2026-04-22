<div align="center">
  
# 🤖 Real-Time AI Face Tracking & Mask Detection Suite
**Enterprise-Grade Computer Vision Application**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Hasnain%20Khan-blueviolet.svg)](https://github.com/Hexecutionerr)

A high-performance dual-system computer vision application. 
Features a **YOLOv8** pipeline for deep-learning Identity Recognition, and a **DNN + Heuristics** pipeline for Mask/Rumal Detection. Built with an Object-Oriented Architecture, hardware optimizations, and robust error handling.

</div>

---

## 🚀 Key Features

* **YOLOv8 Deep Learning Detection:** Auto-downloads a community-trained YOLOv8-Face model for highly accurate, occlusion-resistant face detection.
* **Identity Recognition:** Drop `.jpg` images into a folder, and the system uses `dlib` facial embeddings to match faces to known identities in real time.
* **DNN Mask Detection:** A dedicated module utilizing an SSD ResNet-10 model combined with HSV color analysis and Haar Cascades to detect face masks and cloth coverings.
* **Clean OOP Architecture:** Completely refactored into a scalable structure following software engineering best practices (SOLID principles).
* **Fault-Tolerant System:** Robust `try/except` and Context Managers ensure hardware safely unmounts even during fatal errors.

---

## 📂 Project Architecture

The application is modularized to ensure separation of concerns. **For an in-depth code explanation, please read [ARCHITECTURE_NOTES.md](./ARCHITECTURE_NOTES.md).**

```text
📁 Project Root
│-- 📄 main.py                   # App 1: YOLOv8 Face Tracking & Identity Recognition
│-- 📄 face_mask_detection.py    # App 2: DNN Mask & Cloth Occlusion Detection
│-- 📄 config.py                 # Centralized configuration & logging logic
│-- 📄 requirements.txt          # Python dependencies
│-- 📁 detector/
│    └── 📄 face_detector.py     # YOLOv8 Inference & Dlib recognizer integration
│-- 📁 utils/
│    └── 📄 camera.py            # OOP context manager for hardware lifecycles
│-- 📁 known_faces/              # Dynamic database for identity embeddings (.jpg files)
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hexecutionerr/-Real-Time-AI-Face-Tracking-Mask-Detection-Suite.git
   cd Real-Time-Face-Detection-master
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note for Windows Users:** The identity recognition relies on `dlib`. If installation fails, you must install **CMake** and the **Desktop development with C++** workload via the Visual Studio Installer.

---

## 💡 Usage Guide

### Application 1: Face Tracking & Identity Recognition (YOLOv8)
Run the main entry script:
```bash
python main.py
```
* **To Enable Identity Recognition:** A folder named `known_faces/` is automatically generated. Place a clear headshot image inside (e.g., `Elon_Musk.jpg`). Restart the script. Recognized bounding boxes turn **Magenta**, while unknowns turn **Red**.

### Application 2: Face Mask & Cloth Detection (DNN)
Run the dedicated mask detection script:
```bash
python face_mask_detection.py
```
* Uses deep learning and color histograms to detect if the lower half of the face is covered by a surgical mask, rumal, or cloth.

*(Press `q` or `ESC` to exit any video stream gracefully).*

---

## 🔮 Future Roadmap (Web App Integration)

Because the architecture is fully decoupled, this project is ready to be ported to the web. 
Using **Streamlit**, you can transform `main.py` into a web application in under 20 lines of code!

---

## 👨‍💻 Author

**Hasnain Khan**
* GitHub: [@Hexecutionerr](https://github.com/Hexecutionerr)

---

## 📄 License & Copyright

Copyright (c) 2026 Hasnain Khan.

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details. 
You are free to use, modify, and distribute this software, provided that you give appropriate credit and include the original copyright notice.
