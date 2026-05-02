import gradio as gr
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for deployment
import matplotlib.pyplot as plt
from facebp_core import FaceBPDetector
import time
import io

# Initialize detector
detector = FaceBPDetector()

def process_video(image, age, gender, mode, state):
    if image is None:
        return None, "-- bpm", "--/--", "Camera not started", None
    
    # Image from Gradio is RGB
    # Convert to BGR for detector (OpenCV expects BGR)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process frame
    frame, hr, bp, debug_status, pulse_trace = detector.process_frame(frame, age, gender)
    
    # Handle Scan Mode
    scan_status = debug_status
    if mode == "15s Scan":
        if not state['scanning'] and not state['complete']:
            state['scanning'] = True
            state['start_time'] = time.time()
            scan_status = "Scanning... (0s)"
        elif state['scanning']:
            elapsed = time.time() - state['start_time']
            if elapsed >= 15:
                state['scanning'] = False
                state['complete'] = True
                scan_status = "Scan Complete!"
            else:
                scan_status = f"Scanning... ({int(elapsed)}s)"
        elif state['complete']:
            scan_status = "Scan Complete! Press reset to scan again."
    else:
        state['scanning'] = False
        state['complete'] = False
        scan_status = "Monitoring Live"

    # Convert back to RGB for display
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Format display values
    if hr == 0:
        hr_text = "-- bpm"
    else:
        hr_text = f"{hr} bpm"
    
    if bp[0] == 0:
        bp_text = "--/--"
    else:
        bp_text = f"{bp[0]}/{bp[1]}"
    
    # Create Pulse Plot - Always show something
    fig, ax = plt.subplots(figsize=(3, 1.5))
    
    if pulse_trace and any(x != 0.5 for x in pulse_trace):  # Real signal
        ax.plot(pulse_trace, color='#ff0000', linewidth=1)
        ax.set_title("Live Pulse Signal", fontsize=8, color='green')
    else:  # Flat line or no signal
        ax.plot(pulse_trace if pulse_trace else [0.5]*30, color='#cccccc', linewidth=1)
        ax.set_title("Waiting for Signal...", fontsize=8, color='gray')
    
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0)
    
    return display_frame, hr_text, bp_text, scan_status, state, fig

def reset_scan(state):
    state['scanning'] = False
    state['complete'] = False
    state['start_time'] = 0
    return state, "Ready"

# Custom CSS for better mobile experience
custom_css = """
.gradio-container { 
    max-width: 1200px !important; 
    margin: 0 auto !important;
}
#video-input { 
    height: 400px !important; 
}
.gr-button {
    background: linear-gradient(45deg, #ff6b6b, #ee5a24) !important;
    border: none !important;
    color: white !important;
}
.gr-form {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Face Vitals AI") as demo:
    state = gr.State({'scanning': False, 'start_time': 0, 'complete': False})
    
    gr.Markdown("""
    # 🩸 Face Vitals AI - Real-time HR & BP Monitor
    
    **Real-time heart rate and blood pressure monitoring using your camera**
    
    📱 **Instructions:** Allow camera access → Position face in view → Get instant readings
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            input_video = gr.Image(
                sources=["webcam"], 
                streaming=True, 
                label="📹 Live Video Feed", 
                elem_id="video-input"
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Vital Signs")
            
            with gr.Group():
                hr_display = gr.Text(
                    label="❤️ Heart Rate", 
                    value="-- bpm", 
                    interactive=False,
                    elem_classes="vital-display"
                )
                bp_display = gr.Text(
                    label="🩸 Blood Pressure", 
                    value="--/--", 
                    interactive=False,
                    elem_classes="vital-display"
                )
                status_display = gr.Text(
                    label="📡 System Status", 
                    value="Camera not started", 
                    interactive=False
                )
                pulse_plot = gr.Plot(label="📈 Live Pulse Signal")
            
            gr.Markdown("### ⚙️ Settings")
            with gr.Group():
                age_input = gr.Slider(
                    minimum=18, 
                    maximum=80, 
                    value=25, 
                    label="👤 Age"
                )
                gender_input = gr.Radio(
                    choices=["Male", "Female"], 
                    value="Male", 
                    label="⚧ Gender"
                )
                mode_input = gr.Radio(
                    choices=["Start Monitoring", "15s Scan"], 
                    value="Start Monitoring", 
                    label="🔄 Mode"
                )
                reset_btn = gr.Button("🔄 Reset Scan", variant="secondary")

    gr.Markdown("""
    ### 📋 Normal Ranges
    - **Heart Rate:** 70-90 BPM (resting)
    - **Blood Pressure:** 110-140/70-90 mmHg
    
    ### ⚠️ Disclaimer
    This is for educational purposes only. Not for medical diagnosis.
    """)

    # Real-time processing
    input_video.stream(
        fn=process_video,
        inputs=[input_video, age_input, gender_input, mode_input, state],
        outputs=[input_video, hr_display, bp_display, status_display, state, pulse_plot],
        stream_every=0.1  # Update every 100ms for smooth experience
    )
    
    reset_btn.click(
        fn=reset_scan, 
        inputs=[state], 
        outputs=[state, status_display]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,  # Create public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )