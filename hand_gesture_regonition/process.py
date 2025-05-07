from abc import ABC, abstractmethod
import cv2


class Process(ABC):
    def draw(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        return frame
        
    def process(self):
        pass
    
    def close(self):
        pass
    
    @property
    def program(self):
        from hand_gesture_regonition.program import Program
        return Program.get()