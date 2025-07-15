# Hand Gesture Recognition System

## Goal
This project aims to develop a robust hand gesture recognition system capable of detecting various hand gestures. Building upon the success of its predecessor, this version focuses on enhanced model training, live configuration, and streamlined data collection.

## Key Features
*   **Gesture Recognition:** Recognizes a variety of hand gestures (e.g., left hand, left index, OK, peace, thumb up/down for both hands).
*   **Improved Training:** Features advanced model training capabilities for higher accuracy.
*   **Live Configuration:** Allows for real-time adjustment of system parameters.
*   **Efficient Data Collection:** Provides tools for streamlined data acquisition for model training.

## Screenshots

### Model Training and Metrics
![Model Training and Metrics](assets/Screenshot%20from%202025-07-15%2011-25-45.png)
*Description: An overview of the model training process, displaying key metrics and performance indicators.*

### Live Configuration
![Live Configuration](assets/Screenshot%20from%202025-07-15%2011-26-44.png)
*Description: Interface for real-time configuration and adjustment of system settings.*

### Data Collection
![Data Collection](assets/Screenshot%20from%202025-07-15%2011-27-11.png)
*Description: Tools and interface for collecting and managing gesture data for training.*

## Usage
The project uses Poetry for dependency management.

To run the application:
`poetry run python hand_gesture_regonition`

To train a model:
`poetry run python hand_gesture_regonition/v2/model/cli.py --help`