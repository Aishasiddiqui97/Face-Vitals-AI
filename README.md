# 🩸 FaceBP Buddy

A real-time, non-contact Heart Rate (HR) and Blood Pressure (BP) estimation app using rPPG (remote Photoplethysmography).

## 🚀 How to Run

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the App**:
    ```bash
    python app.py
    ```
3.  **Access the Dashboard**:
    Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

## 🛠 Features

-   **Real-time Webcam Feed**: Uses MediaPipe for precise face tracking.
-   **rPPG Signal Processing**: Extracts pulse data from the green channel of your forehead.
-   **BP Estimation**: Uses a heuristic formula combining HR, Age, and Gender.
-   **Scan Mode**: 15-second diagnostic for more stable readings.
-   **Color Alerts**: Visual indicators for normal, borderline, and high/low readings.

## ⚠️ Medical Disclaimer

**This is for demonstration and educational purposes only.**
The readings are NOT clinical-grade and should not be used for medical diagnosis. The formula used for Blood Pressure is a statistical approximation and may not be accurate for all individuals.

## 📦 Requirements

-   Webcam
-   Python 3.8+
-   Libraries: `opencv-python`, `mediapipe`, `numpy`, `scipy`, `gradio`
