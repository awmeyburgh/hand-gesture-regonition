import cv2
from mediapipe.python.solutions import hands, drawing_utils

from hand_gesture_regonition.v1.process import Process


class HandsOverlay(Process):
    def __init__(self, draw_landmarks = True):
        self.draw_landmarks = draw_landmarks
        self.hands = hands.Hands(
            min_detection_confidence=0.8, min_tracking_confidence=0.5
        )
        self.detection = None

    def draw(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self.detection.multi_hand_landmarks and self.draw_landmarks:
            for hand_lms in self.detection.multi_hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_lms,
                    hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(
                        color=(255, 0, 255), thickness=4, circle_radius=2
                    ),
                    connection_drawing_spec=drawing_utils.DrawingSpec(
                        color=(20, 180, 90), thickness=2, circle_radius=2
                    ),
                )

        return frame

    def process(self):
        self.detection = self.hands.process(self.program.frame)

    def close(self):
        self.hands.close()
