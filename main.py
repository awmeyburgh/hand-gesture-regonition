# from pathlib import Path
# from typing import Optional
# import mediapipe as mp
# import cv2
# from PIL import Image, ImageOps

# from mediapipe.python.solutions import drawing_utils as mp_drawing
# from mediapipe.python.solutions import hands as mp_hands
# import numpy as np

# from hand_gesture_regonition.gesture import Gesture, GestureLibrary, GestureLibraryDataset, GestureVarients

# capture = cv2.VideoCapture(0)

# def draw_border(img: cv2.typing.MatLike, color = "green", thickness = 10) -> cv2.typing.MatLike:
# 	pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 	result_pil_img = ImageOps.expand(pil_img, border=thickness, fill=color)

# 	return cv2.cvtColor(np.array(result_pil_img), cv2.COLOR_RGB2BGR)

# recording = False
# gesture_library = GestureLibrary('data')
# gesture_varients = gesture_library.get('wave_left')
# gesture: Optional[Gesture] = None

# with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
# 	while capture.isOpened():
# 		ret, frame = capture.read()
# 		frame = cv2.flip(frame, 1)
# 		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		detected_image = hands.process(image)
# 		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	
# 		if detected_image.multi_hand_landmarks:
# 			for hand_lms in detected_image.multi_hand_landmarks:
# 				mp_drawing.draw_landmarks(image, hand_lms,
# 											mp_hands.HAND_CONNECTIONS,
# 											landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
# 												color=(255, 0, 255), thickness=4, circle_radius=2),
# 											connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
# 												color=(20, 180, 90), thickness=2, circle_radius=2)
# 											)
# 		if cv2.pollKey() & 0xFF == ord('r'):		
# 			recording = not recording

# 			if recording:
# 				gesture = Gesture()
# 			else:
# 				gesture_varients.save(gesture)
# 				gesture = None

# 		if recording:
# 			image = draw_border(image)
			
# 			if detected_image.multi_hand_landmarks:
# 				gesture.capture(detected_image)
	
# 		cv2.imshow('Webcam', image)
	
# 		if cv2.pollKey() & 0xFF == ord('q'):
# 			break

# capture.release()
# cv2.destroyAllWindows()

from hand_gesture_regonition import main

if __name__ == '__main__':
	main()