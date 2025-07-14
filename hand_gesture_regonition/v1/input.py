import cv2

from hand_gesture_regonition.v1.process import Process


class Input(Process):
    def __init__(self):
        self.key = None
    
    def process(self):
        self.key = cv2.pollKey()
    
    @classmethod
    def get(cls) -> "Input":
        from hand_gesture_regonition.v1.program import Program
        return Program.get().input
    
    @classmethod
    def is_pressed(cls, key) -> bool:
        self = cls.get()
        
        if self.key is None:
            return False
        return self.key & 0xFF == ord(key)