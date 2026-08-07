Markdown
# 📷 CamVision AI — Real-Time Webcam Image Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-SVM-orange.svg)](https://scikit-learn.org/)

**CamVision AI** — Veb-kamera orqali real vaqt rejimida obyektlar va qo'l harakatlarini (gestures) aniqlash hamda klassifikatsiya qilish uchun mo'ljallangan Kompyuter Ko'rishi (Computer Vision) va Mashinali O'rgatish (Machine Learning) ilovasi.

Foydalanuvchilar Tkinter interfeysi orqali o'zlari xohlagan 2 ta sinf (class) uchun kadrlar to'plamini (dataset) yig'ishlari va joyida **LinearSVC (Support Vector Machine)** modelini o'qitib, darhol natijani sinab ko'rishlari mumkin.

---

## 🌟 Asosiy Xususiyatlari

* **Real-time Video Feed:** Veb-kamera orqali uzluksiz tasvir uzatish va ishlov berish.
* **Interaktiv Dataset Yaratish:** Tugmani bir martalik bosish orqali rasmlarni avtomatik o'lchamlarga keltirib saqlash.
* **On-the-Fly ML O'qitish:** Scikit-Learn `LinearSVC` modeli yordamida yig'ilgan kadrlar to'plamini bir necha soniyada o'qitish.
* **Auto Prediction:** O'qitilgan model yordamida real vaqt rejimida uzluksiz klassifikatsiya qilish.
* **Qulay GUI:** Tkinter va Pillow (PIL) kutubxonalari asosida yaratilgan sodda va intuitiv interfeys.

---

## 🛠️ Texnologiyalar (Tech Stack)

* **Dasturlash tili:** Python 3.x
* **Computer Vision:** OpenCV (`opencv-python`)
* **Machine Learning:** Scikit-Learn (`scikit-learn`)
* **GUI / Interfeys:** Tkinter, Pillow (`PIL`)
* **Ma'lumotlar bilan ishlash:** NumPy

---

## 📂 Loyiha Strukturasi

```text
realtime-cam-classifier/
├── app.py          # Tkinter interfeysi va grafik mantiq
├── camera.py       # OpenCV orqali veb-kamberani boshqarish
├── model.py        # Machine Learning (LinearSVC) moduli
├── main.py         # Dasturni ishga tushirish nuqtasi
├── requirements.txt# Loyiha kutubxonalari ro'yxati
├── .gitignore      # Git xotirasidan chiqarib tashlanadigan fayllar
└── README.md       # Loyiha hujjati
🚀 O'rnatish va Ishga Tushirish
1. Repozitoriyani klonlash:
Bash
git clone [https://github.com/Asrorbek0002/realtime-cam-classifier.git](https://github.com/Asrorbek0002/realtime-cam-classifier.git)
cd realtime-cam-classifier
2. Virtual muhitni faollashtirish (tavsiya etiladi):
Bash
python3 -m venv ai_env
source ai_env/bin/activate  # Linux / macOS uchun
3. Kerakli kutubxonalarni o'rnatish:
Bash
pip install -r requirements.txt
4. Dasturni ishga tushirish:
Bash
python main.py
🎮 Dasturdan Foydalanish Yo'riqnomasi
Sinf nomlarini kiriting: Dastur ochilganda dialoq oynasiga 2 ta sinf nomini yozing (masalan: Ruchka va Daftar yoki Qo'l Ochiq va Qo'l Yopiq).

Dataset yig'ing: Kameraga obyektni ko'rsatib, 1-sinf va 2-sinf tugmalarini har bir sinf uchun kamida 15-20 marta bosing.

Modelni o'qiting: Train Model tugmasini bosing. Model tayyor bo'lganida tasdiqlovchi xabar chiqadi.

Bashorat qiling:

Bir martalik aniqlash uchun Predict tugmasini bosing.

Real vaqt rejimida uzluksiz aniqlash uchun Auto Prediction rejimini yoqing.

Tozalash: Yangidan boshlash uchun Reset tugmasini bosing.
