from typing import List, Optional
import cv2
import numpy as np

from hand_gesture_regonition.gesture import GestureLibrary
from hand_gesture_regonition.gesture_recorder import GestureRecorder
from hand_gesture_regonition.gesture_regonition import GestureCommands, GestureRegonition
from hand_gesture_regonition.hands_overlay import HandsOverlay
from hand_gesture_regonition.input import Input
from hand_gesture_regonition.network import GestureRegonitionNetwork
from hand_gesture_regonition.process import Process


class Program:
    __SINGLETON = None
    
    @classmethod
    def get(cls) -> "Program":
        if cls.__SINGLETON is None:
            cls.__SINGLETON = Program()
            
        return cls.__SINGLETON
    
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.frame: Optional[cv2.typing.MatLike] = None
        
        self.input = Input()
        self.hands_overlay = HandsOverlay()
        self.gesture_library = GestureLibrary('data')
        
        self.processes: List[Process] = [
            self.input,
            self.hands_overlay,
            GestureRecorder(self.gesture_library),
            GestureRegonition("network.bin")
        ]

    def close(self):
        self.capture.release()
        
        for process in self.processes:
            process.close()
        
        cv2.destroyAllWindows()

    def stop(self):
        self.capture.release()

    def is_running(self) -> bool:
        return self.capture.isOpened()
    
    def capture_frame(self) -> cv2.typing.MatLike:
        _, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def process(self):
        for process in self.processes:
            process.process()
        
        if Input.is_pressed('q'):
            self.stop()

    def draw(self):
        for process in self.processes:
            self.frame = process.draw(self.frame)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Webcam', self.frame)

    def run(self):
        while self.is_running():
            self.frame = self.capture_frame()

            self.process()
            self.draw()

        self.close()

