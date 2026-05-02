from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import json
from facebp_core import FaceBPDetector
import io
from PIL import Image

app = Flask(__name__)
detector = FaceBPDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        pil_image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Get user settings
        age = data.get('age', 25)
        gender = data.get('gender', 'Male')
        
        # Process frame
        processed_frame, hr, bp, status, pulse_trace = detector.process_frame(frame, age, gender)
        
        # Convert processed frame back to base64
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        pil_processed = Image.fromarray(processed_rgb)
        buffer = io.BytesIO()
        pil_processed.save(buffer, format='JPEG')
        processed_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'heart_rate': hr,
            'blood_pressure': bp,
            'status': status,
            'pulse_trace': pulse_trace,
            'processed_image': f"data:image/jpeg;base64,{processed_b64}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)