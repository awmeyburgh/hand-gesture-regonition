from typing import Optional

import cv2
from hand_gesture_regonition.gesture import Gesture, GestureLibrary, GestureVarients
from hand_gesture_regonition.input import Input
from hand_gesture_regonition.process import Process


class GestureRecorder(Process):
    BORDER_WIDTH = 10
    BORDER_COLOR = (0, 255, 0)

    def __init__(self, library: GestureLibrary, enabled=True):
        self.library = library
        self.key = 0
        self.gesture: Optional[Gesture] = None
        self.recording = False
        self.enabled = enabled

    def draw(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self.enabled:
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
                
            frame = cv2.putText(
                frame, 
                self.variants.name,
                (5, 20),
                cv2.QT_FONT_NORMAL,
                0.5,
                (0, 0, 0)
            )

        return frame
    
    @property
    def variants(self):
        return self.library.get(self.library.keys()[self.key])

    def process(self):
        if self.enabled:
            if Input.is_pressed("r"):
                self.recording = not self.recording

                if self.recording:
                    detection = self.program.hands_overlay.detection
                    
                    if detection.multi_handedness:
                        self.gesture = Gesture(is_right=detection.multi_handedness[0].classification[0].index == 1)
                        
                        if self.variants.is_static:
                            self.gesture.capture(detection)
                            self.variants.save(self.gesture)
                            self.gesture = None
                            self.recording = False
                else:
                    self.variants.save(self.gesture)
                    self.gesture = None
                    
            if Input.is_pressed("d"):
                self.key = (self.key + 1) % len(self.library.keys())
                
            if Input.is_pressed("a"):
                self.key = (self.key - 1 + len(self.library.keys())) % len(self.library.keys())

            if self.recording:
                self.gesture.capture(self.program.hands_overlay.detection)
