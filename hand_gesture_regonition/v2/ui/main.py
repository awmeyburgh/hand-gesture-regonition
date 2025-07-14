from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import os
from pathlib import Path
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    # Decode the image
    image_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image with MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for landmark in hand_landmarks.landmark:
                landmark_list.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
            landmarks.append(landmark_list)

    # Send the landmarks back to the client
    emit('landmarks', landmarks)

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    gesture = data['gesture']
    image_bytes = base64.b64decode(image_data)

    # Create gesture-specific directory
    gesture_dir = Path('data/v2/static') / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    # Generate a filename with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'{timestamp}.png'
    filepath = gesture_dir / filename

    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    return jsonify({'success': True, 'filepath': filepath})

@app.route('/record', methods=['POST'])
def record():
    video_file = request.files['video']
    gesture = request.form['gesture']
    
    # Create a unique directory for the recording
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    record_dir = Path('data/v2/sequential') / gesture / f'{timestamp}'
    record_dir.mkdir(parents=True, exist_ok=True)

    # Save the video file temporarily
    temp_video_path = record_dir / 'temp_video.webm'
    video_file.save(temp_video_path)

    # Use OpenCV to extract frames
    cap = cv2.VideoCapture(str(temp_video_path))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        frame_filename = record_dir / f'frame_{frame_count:04d}.png'
        cv2.imwrite(str(frame_filename), frame)
        frame_count += 1

    cap.release()
    temp_video_path.unlink() # Clean up the temporary video file

    return jsonify({'success': True, 'message': f'Saved {frame_count} frames to {record_dir}'})

def run(debug=True):
    socketio.run(app, debug=debug)