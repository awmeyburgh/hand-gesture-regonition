from abc import ABC
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from hand_gesture_regonition.gesture import Gesture, GestureLibrary
from hand_gesture_regonition.gesture_commands import GestureCommands
from hand_gesture_regonition.home_assistant_commands import HomeAssistantCommands
from hand_gesture_regonition.network import StaticGRNetwork
from hand_gesture_regonition.process import Process
        
    
class GestureRegonition(Process):
    def __init__(self, confidence=0.8, delta_threshold=0.5):
        self.static_file = Path("static_network.bin")
        self.static_modal = StaticGRNetwork.load(self.static_file) if self.static_file.exists() else None
        self.commands = self.create_commands()
        self.condidence = confidence
        self.gestures = [Gesture(is_right=False), Gesture(is_right=True)]
        self.captured = []
        self.executor = ThreadPoolExecutor(1)
        self.future = None
        self.delta_threshold = delta_threshold
        self.last_keys = [None, None]
        
    def static_forward(self):
        if self.static_modal:
            return [self.static_modal.classify(self.gestures[i]) if self.captured[i] else None for i in range(len(self.gestures))]
        return [None, None]
            
    def forward(self):
        keys = self.static_forward()
        
        for i in range(2):
            if keys[i] != self.last_keys[i]:
                self.last_keys[i] = keys[i]
                if keys[i] is not None:
                    self.commands.call(keys[i])
        
    def capture(self) -> bool:
        detection = self.program.hands_overlay.detection
        
        self.captured = [g.capture(detection) for g in self.gestures]
        
        for i in range(2):
            if not self.captured[i]:
                self.last_keys[i] = None
        
        return np.any(self.captured)
        
    def process(self):
        if self.static_modal is not None:
            if self.capture():
                if self.future is None:
                    self.future = self.executor.submit(self.forward)
                
            if self.future is not None and self.future.done():
                self.future = None
            
    def draw(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        keys = [k for k in self.last_keys if k is not None]
        if len(keys) > 0:
            frame = cv2.putText(
                frame, 
                ', '.join(keys),
                (frame.shape[0]-160, 20),
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