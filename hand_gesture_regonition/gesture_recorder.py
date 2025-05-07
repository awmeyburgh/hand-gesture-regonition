from typing import Optional

import cv2
from hand_gesture_regonition.gesture import Gesture, GestureVarients
from hand_gesture_regonition.input import Input
from hand_gesture_regonition.process import Process


class GestureRecorder(Process):
    BORDER_WIDTH = 10
    BORDER_COLOR = (0, 255, 0)

    def __init__(self, variants: GestureVarients):
        self.variants = variants
        self.gesture: Optional[Gesture] = None
        self.recording = False

    def draw(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self.recording:
            frame = cv2.copyMakeBorder(
                frame,
                self.BORDER_WIDTH,
                self.BORDER_WIDTH,
                self.BORDER_WIDTH,
                self.BORDER_WIDTH,
                cv2.BORDER_CONSTANT,
                value=self.BORDER_COLOR,
            )

        return frame

    def process(self):
        if Input.is_pressed("r"):
            self.recording = not self.recording

            if self.recording:
                self.gesture = Gesture()
            else:
                self.variants.save(self.gesture)
                self.gesture = None

        if self.recording:
            detection = self.program.hands_overlay.detection

            if detection.multi_hand_landmarks:
                self.gesture.capture(detection)
