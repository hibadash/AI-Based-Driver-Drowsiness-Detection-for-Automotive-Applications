# Drowsiness Detection for Drivers 

## Overview
This project aims to enhance road safety by detecting driver drowsiness in real-time using computer vision techniques.  
The system monitors the driver’s face and eyes via a webcam, analyzing blink rate and eye closure to detect signs of fatigue. Alerts are triggered when drowsiness is detected.

---

## Features
- Real-time detection of driver drowsiness
- Eye aspect ratio (EAR) calculation to detect prolonged eye closure
- Audio or visual alert when drowsiness is detected
- Optional logging of drowsiness events for analysis

---

## Motivation
Road accidents caused by driver fatigue are a major safety concern in the automotive field.  
This project demonstrates how AI and computer vision can be applied to **improve driver safety**.

---

## Technologies Used
- **Python** – Core programming language
- **OpenCV** – Real-time image processing
- **dlib / Mediapipe** – Facial landmark detection
- **NumPy / Pandas** – Data handling and analysis
- **Streamlit / Tkinter** (optional) – GUI for real-time monitoring

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
