OralGuard AI — Early Oral Cancer Detection

An AI-powered web application that detects oral cancer from images using deep learning and provides visual explanations with Grad-CAM.

---

Features

 AI-based image classification (MobileNetV2)
 Grad-CAM heatmap visualization
 PDF clinical report generation
 Camera capture & image upload
 Analysis history tracking
 Nearby hospital finder
 Dark mode support

---

Tech Stack

- Backend: Django, Django REST Framework  
- AI Model: TensorFlow, Keras (MobileNetV2)  
- Computer Vision: OpenCV  
- Frontend: HTML, CSS, JavaScript  
- Database: SQLite  

---

Setup

python -m venv env

env\Scripts\activate

pip install -r requirements.txt

python manage.py migrate

python manage.py runserver
