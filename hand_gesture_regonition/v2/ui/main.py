from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import os
from pathlib import Path
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import yaml
import subprocess
from hand_gesture_regonition.v2.dataset import load_dataset, static as static_dataset
from hand_gesture_regonition.v2.dataset import sequential as sequential_dataset
from hand_gesture_regonition.v2.model import load_model

loaded_models = {}

app = Flask(__name__)
socketio = SocketIO(app)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


CONFIG_PATH = Path('data/v2/model/config.yaml')

def load_model_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models/config', methods=['GET'])
def get_models_config():
    config = load_model_config()
    return jsonify(config)


@socketio.on('image')
def handle_image(image_data_str, extra_data):
    # Decode the image
    image_data = base64.b64decode(image_data_str.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image with MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    landmarks = []
    handedness_list = []
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmark_list = []
            for landmark in hand_landmarks.landmark:
                landmark_list.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
            landmarks.append(landmark_list)
            handedness_list.append(results.multi_handedness[i].classification[0].label)

    predictions_list = []

    try:
        model_key: str = extra_data.get('model_key')
        min_confidence: float = extra_data.get('min_confidence', 0) / 100.0  # Default to 0 if not provided

        if model_key and model_key not in loaded_models:
            loaded_models[model_key] = {
                'model': load_model(model_key),
                'labels': load_dataset(model_key[:model_key.index('/')])[4]
            }

        if model_key and model_key in loaded_models:
            model = loaded_models[model_key]['model']
            labels = loaded_models[model_key]['labels']
            if landmarks:
                for hand_landmarks_list in landmarks:
                    # Flatten the landmarks for the current hand
                    flat_landmarks_for_hand = [coord for landmark in hand_landmarks_list for coord in [landmark['x'], landmark['y'], landmark['z']]]
                    
                    # Predict gesture and confidence for the current hand
                    y_pred_proba = model.predict([flat_landmarks_for_hand])[0]
                    predicted_index = np.argmax(y_pred_proba)
                    gesture = labels[predicted_index]
                    confidence = y_pred_proba[predicted_index]
                    
                    if confidence >= min_confidence:
                        predictions_list.append({'gesture': gesture, 'confidence': confidence})
    except Exception as e:
        print(e)
            

    # Send the landmarks and predictions back to the client
    emit('landmarks', {'landmarks': landmarks, 'predictions': predictions_list})

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

@socketio.on('compile_static_dataset')
def compile_static_dataset():
    try:
        df = static_dataset.generate()
        static_dataset.save(df)
    except Exception as e:
        print(f"Error compiling static dataset: {e}") # Log to server console for debugging

@socketio.on('compile_sequential_dataset')
def compile_sequential_dataset():
    try:
        df = sequential_dataset.generate()
        sequential_dataset.save(df)
        emit('compile_sequential_complete')
    except Exception as e:
        print(f"Error compiling sequential dataset: {e}") # Log to server console for debugging
        emit('compile_sequential_complete') # Ensure spinner is hidden even on error

@socketio.on('run_model_cli')
def run_model_cli(data):
    model_key = data['model_key']
    save_model = data['save_model']
    load_model = data['load_model']
    train = data['train']
    test = data['test']
    train_size = data['train_size']
    hyperparameters = data['hyperparameters']

    command = ["poetry", "run", "python", "hand_gesture_regonition/v2/model/cli.py", model_key]

    command.extend(["--train-size", str(train_size)])

    if save_model:
        command.append("--save")
    if load_model:
        command.append("--load")
    if train:
        command.append("--train")
    if test:
        command.append("--test")

    # Collect hyperparameters as a list of strings for the 'params' argument
    cli_params = ['--']
    if hyperparameters:
        for key, value in hyperparameters.items():
            cli_params.append(key)
            str_value = str(value)
            # Quote values that start with a hyphen to prevent them from being interpreted as options
            if str_value.startswith('-'):
                cli_params.append(f'"{str_value}"')
            else:
                cli_params.append(str_value)

    command.extend(cli_params)

    socketio.emit('clear_console')
    socketio.emit('console_output', {'output': f"Running command: {' '.join(command)}\n"})

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            socketio.emit('console_output', {'output': line})
        process.stdout.close()
        return_code = process.wait()
        socketio.emit('console_output', {'output': f"Command finished with exit code {return_code}\n"})

        if (train or test) and return_code == 0:
            # Invalidate the cached model so the new one is loaded on next prediction
            if model_key in loaded_models:
                del loaded_models[model_key]
            try:
                with open(CONFIG_PATH.parent / "results.yaml", 'r') as f:
                    metrics_data = yaml.safe_load(f)
                socketio.emit('metrics_update', metrics_data)
            except FileNotFoundError:
                socketio.emit('console_output', {'output': "results.yaml not found. Metrics could not be loaded.\n"})
            except Exception as e:
                socketio.emit('console_output', {'output': f"Error loading metrics: {e}\n"})

    except Exception as e:
        socketio.emit('console_output', {'output': f"Error running command: {e}\n"})

def run(debug=True):
    socketio.run(app, debug=debug)