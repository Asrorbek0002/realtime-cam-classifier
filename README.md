# 📷 CamVision AI — Real-Time Webcam Image Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-SVM-orange.svg)](https://scikit-learn.org/)

**CamVision AI** is a real-time Computer Vision and Machine Learning application designed to classify objects and hand gestures live through a webcam feed. 

Built with **Tkinter** and **Scikit-Learn**, it allows users to dynamically capture custom image datasets for two separate classes, train a **LinearSVC (Support Vector Machine)** model on the fly, and perform real-time automated predictions.

---

## 🌟 Features

* **Real-Time Video Feed:** Continuous low-latency webcam frame streaming and processing.
* **Interactive Dataset Collection:** Instant dataset gathering with automated image resizing and grayscale conversion.
* **On-the-Fly ML Training:** Train a custom `LinearSVC` classification model within seconds directly from the GUI.
* **Auto Prediction:** Live, automated continuous prediction mode for real-time gesture/object recognition.
* **Clean GUI:** User-friendly interface built using Tkinter and Pillow (PIL).

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3.x
* **Computer Vision:** OpenCV (`opencv-python`)
* **Machine Learning:** Scikit-Learn (`scikit-learn`)
* **GUI / Interface:** Tkinter, Pillow (`PIL`)
* **Data Processing:** NumPy

---

## 📂 Project Structure

```text
realtime-cam-classifier/
├── app.py          # Tkinter GUI logic and application flow
├── camera.py       # OpenCV webcam input handler
├── model.py        # Machine Learning (LinearSVC) model wrapper
├── main.py         # Application entry point
├── requirements.txt# Project dependencies
├── .gitignore      # Git ignore rules for cached/temporary files
└── README.md       # Project documentation
🚀 Installation & Setup
1. Clone the repository:
Bash
git clone [https://github.com/Asrorbek0002/realtime-cam-classifier.git](https://github.com/Asrorbek0002/realtime-cam-classifier.git)
cd realtime-cam-classifier
2. Create and activate a virtual environment (recommended):
Bash
python3 -m venv ai_env
source ai_env/bin/activate  # On Linux / macOS
# ai_env\Scripts\activate   # On Windows
3. Install dependencies:
Bash
pip install -r requirements.txt
4. Run the application:
Bash
python main.py
🎮 How to Use
Set Class Names: Upon launching, enter custom names for your two target classes (e.g., Pen vs Notebook or Open Hand vs Fist).

Collect Dataset: Hold an object in front of the camera and click the class buttons to capture 15–20 sample frames per class.

Train the Model: Click Train Model. A confirmation popup will notify you once training is complete.

Predict:

Click Predict for single-frame classification.

Toggle Auto Prediction for continuous real-time recognition.

Reset: Click Reset anytime to clear all collected image data and model states.
