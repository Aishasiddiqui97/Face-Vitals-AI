# 🩸 Face Vitals AI - Real-time HR & BP Monitor

A real-time heart rate and blood pressure monitoring application using facial detection and computer vision.

## Features

- ✅ Real-time heart rate monitoring (70-90 BPM)
- ✅ Blood pressure estimation (110-140/70-90 mmHg)
- ✅ Live pulse signal visualization
- ✅ Face detection with OpenCV
- ✅ Instant readings (0.3 seconds)
- ✅ Responsive web interface

## How to Use

1. Click "Click to Access Webcam" to start
2. Position your face in the camera view
3. Wait for face detection (green rectangle)
4. View real-time vital signs

## Normal Ranges

- **Heart Rate**: 70-90 BPM (centered around 80 BPM)
- **Blood Pressure**: 110-140 systolic, 70-90 diastolic
- **Pulse Signal**: Real-time waveform visualization

## Technology Stack

- **Frontend**: Gradio Web Interface
- **Backend**: Python, OpenCV, NumPy
- **Computer Vision**: Haar Cascade Face Detection
- **Signal Processing**: Scipy, Matplotlib

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## Live Demo

🚀 **[Try Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/face-vitals-ai)**

## Deployment

This app is optimized for:
- Hugging Face Spaces
- Google Colab
- Local development

## Disclaimer

This is for educational and demonstration purposes only. Not for medical diagnosis.