import streamlit as st
import cv2
import numpy as np
from facebp_core import FaceBPDetector
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Face Vitals AI",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Face Vitals AI - Real-time HR & BP Monitor")

# Initialize detector
if 'detector' not in st.session_state:
    st.session_state.detector = FaceBPDetector()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Camera Feed")
    camera = st.camera_input("Take a photo for vital signs")

with col2:
    st.subheader("📊 Vital Signs")
    hr_placeholder = st.empty()
    bp_placeholder = st.empty()
    status_placeholder = st.empty()
    
    st.subheader("⚙️ Settings")
    age = st.slider("Age", 18, 80, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])

if camera:
    # Process image
    image = cv2.imdecode(np.frombuffer(camera.read(), np.uint8), cv2.IMREAD_COLOR)
    frame, hr, bp, status, pulse_trace = st.session_state.detector.process_frame(image, age, gender)
    
    # Display results
    hr_placeholder.metric("❤️ Heart Rate", f"{hr} bpm" if hr > 0 else "-- bpm")
    bp_placeholder.metric("🩸 Blood Pressure", f"{bp[0]}/{bp[1]}" if bp[0] > 0 else "--/--")
    status_placeholder.info(f"📡 Status: {status}")
    
    # Show processed frame
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Feed")

st.markdown("""
### 📋 Normal Ranges
- **Heart Rate:** 70-90 BPM
- **Blood Pressure:** 110-140/70-90 mmHg

⚠️ **Disclaimer:** Educational purposes only. Not for medical diagnosis.
""")