from abc import ABC
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import cv2
import torch

from hand_gesture_regonition.gesture import Gesture, GestureLibrary
from hand_gesture_regonition.gesture_commands import GestureCommands
from hand_gesture_regonition.home_assistant_commands import HomeAssistantCommands
from hand_gesture_regonition.network import GestureRegonitionNetwork
from hand_gesture_regonition.process import Process
        
    
class GestureRegonition(Process):
    def __init__(self, file, confidence=0.8, delta_threshold=0.5):
        self.file = Path(file)
        self.modal = GestureRegonitionNetwork.load(self.file) if self.file.exists() else None
        self.commands = self.create_commands()
        self.condidence = confidence
        self.gesture = Gesture()
        self.executor = ThreadPoolExecutor(1)
        self.future = None
        self.delta_threshold = delta_threshold
        self.last_key = None
        
    def forward(self):
        x = self.gesture.tensor
        x = x[None, :, :]
        y_pred = self.modal.forward(x)
        
        y_pred = y_pred >= self.condidence
        
        if y_pred.sum().item() == 1:
            _, index = torch.max(y_pred, 1)
            key = GestureLibrary.keys()[index.item()]
            if key != self.last_key:
                self.last_key = key
                self.commands.call(key)
        
    def process(self):
        if self.modal is not None:
            if self.program.hands_overlay.detection.multi_hand_landmarks:
                self.gesture.capture(self.program.hands_overlay.detection)
                
                if self.future is None:
                    self.future = self.executor.submit(self.forward)
            else:
                self.gesture.frames = []
                
            if self.future is not None and self.future.done():
                self.future = None
            
    def draw(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self.last_key:
            frame = cv2.putText(
                frame, 
                self.last_key,
                (frame.shape[0]-20, 20),
                cv2.QT_FONT_NORMAL,
                0.5,
                (0, 0, 0)
            )
        return frame
            
    def close(self):
        self.commands.close()
        
    def create_commands(self):
        if os.environ.get('HOME_ASSISTANT_COMMANDS', 'False').lower() == 'true':
            return HomeAssistantCommands()
        return GestureCommands()