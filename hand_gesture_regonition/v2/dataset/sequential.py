import pandas as pd
import cv2
import mediapipe as mp
from pathlib import Path

DATASET_PATH = Path("data/v2/sequential/dataset.csv")
IMAGE_DIR = Path("data/v2/sequential/")

def load():
    """Loads the dataset from the specified CSV file."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()

def save(df):
    """Saves the DataFrame to the specified CSV file."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

def generate():
    """
    Generates a DataFrame with landmark data from images, including a frame number.
    The DataFrame will have columns: name, gesture, frame, and landmark columns.
    Assumes data is organized as: IMAGE_DIR/gesture_name/sequence_id/frame_XXX.png
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    data = []
    
    # Iterate through gesture subdirectories
    for gesture_path in IMAGE_DIR.iterdir():
        if gesture_path.is_dir():
            gesture_name = gesture_path.name
            # Iterate through sequence subdirectories within each gesture
            for sequence_path in gesture_path.iterdir():
                if sequence_path.is_dir():
                    sequence_id = sequence_path.name
                    for image_path in sorted(sequence_path.iterdir()): # Sort to ensure frame order
                        if image_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                            filename = image_path.name
                            
                            # Extract frame number from filename (assuming format like frame_001.png)
                            try:
                                frame_number = int(filename.split('_')[-1].split('.')[0])
                            except ValueError:
                                print(f"Warning: Could not extract frame number from {filename}. Skipping.")
                                continue

                            image = cv2.imread(str(image_path))
                            if image is None:
                                print(f"Warning: Could not read image {image_path}")
                                continue
                            
                            # Convert the BGR image to RGB.
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            results = hands.process(image_rgb)
                            
                            if results.multi_hand_landmarks and results.multi_handedness:
                                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                    handedness = results.multi_handedness[i].classification[0].label
                                    
                                    process_hand = False
                                    if gesture_name.startswith("left_") and handedness == "Left":
                                        process_hand = True
                                    elif gesture_name.startswith("right_") and handedness == "Right":
                                        process_hand = True
                                    elif not gesture_name.startswith("left_") and not gesture_name.startswith("right_"):
                                        # If gesture name doesn't specify hand, process any detected hand
                                        process_hand = True

                                    if process_hand:
                                        row = {'name': filename, 'gesture': gesture_name, 'sequence_id': sequence_id, 'frame': frame_number}
                                        for j, landmark in enumerate(hand_landmarks.landmark):
                                            row[f'landmark_x_{j}'] = landmark.x
                                            row[f'landmark_y_{j}'] = landmark.y
                                            row[f'landmark_z_{j}'] = landmark.z
                                        data.append(row)
    
    hands.close()
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Example usage:
    # Generate the dataset
    print("Generating sequential dataset...")
    df = generate()
    print(f"Generated {len(df)} rows.")
    
    # Save the dataset
    save(df)
    print("Sequential dataset saved.")
    
    # Load the dataset
    loaded_df = load()
    print(f"Loaded {len(loaded_df)} rows.")
    print(loaded_df.head())
