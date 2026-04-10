import gradio as gr
import cv2
import numpy as np
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
    
    # Image from Gradio is RGB
    # Convert to BGR for detector (OpenCV expects BGR)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Process frame
    frame, hr, bp, debug_status, pulse_trace = detector.process_frame(frame, age, gender)
    
    # Get alert color
    # color_class = detector.get_color_alert(bp, hr) # Not used currently in text
    
    # Handle Scan Mode
    # State: { 'scanning': bool, 'start_time': float, 'complete': bool }
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
    
    # Create Pulse Plot
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(pulse_trace, color='#ff0000', linewidth=2)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    fig.patch.set_alpha(0)
    
    return display_frame, hr_text, bp_text, scan_status, state, fig

def reset_scan(state):
    state['scanning'] = False
    state['complete'] = False
    state['start_time'] = 0
    return state, "Ready"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State({'scanning': False, 'start_time': 0, 'complete': bool(False)})
    
    gr.Markdown("# 🩸 FaceBP Buddy - Real-time HR & BP Monitor")
    
    with gr.Row():
        with gr.Column(scale=4):
            input_video = gr.Image(sources=["webcam"], streaming=True, label="Live Video Feed", elem_id="video-input")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Live Statistics")
            
            with gr.Group():
                hr_display = gr.Text(label="Heart Rate", value="-- bpm", interactive=False)
                bp_display = gr.Text(label="Blood Pressure", value="--/--", interactive=False)
                status_display = gr.Text(label="System Status", value="Camera not started", interactive=False)
                pulse_plot = gr.Plot(label="Live Pulse Signal")
            
            with gr.Group():
                age_input = gr.Slider(minimum=1, maximum=100, value=25, label="User Age")
                gender_input = gr.Radio(choices=["Male", "Female"], value="Male", label="Gender")
                mode_input = gr.Radio(choices=["Start Monitoring", "15s Scan"], value="Start Monitoring", label="Operational Mode", interactive=True)
                reset_btn = gr.Button("Reset / Restart Scan")

    # Real-time processing
    input_video.stream(
        fn=process_video,
        inputs=[input_video, age_input, gender_input, mode_input, state],
        outputs=[input_video, hr_display, bp_display, status_display, state, pulse_plot]
    )
    
    reset_btn.click(fn=reset_scan, inputs=[state], outputs=[state, status_display])

if __name__ == "__main__":
    demo.launch(css=".gradio-container { max-width: 1600px !important; } #video-input { height: 800px !important; }")
