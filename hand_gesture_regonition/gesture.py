from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from numpy import double

@dataclass
class Landmark:
    tensor: torch.Tensor
    offset: int
    
    @property
    def x(self):
        return self.tensor[self.offset].item()
    
    @property
    def y(self):
        return self.tensor[self.offset+1].item()
    
    @property
    def z(self):
        return self.tensor[self.offset+2].item()

@dataclass
class Hand:
    TSIZE = 3*21
    
    tensor: torch.Tensor
    offset: int
    
    @property
    def landmarks(self) -> List[Landmark]:
        return [Landmark(self.tensor, self.offset+3*i) for i in range(21)]
            

@dataclass
class GestureFrame:
    TSIZE = 2*Hand.TSIZE
    
    tensor: torch.Tensor
    
    @property
    def left(self) -> Hand:
        return Hand(self.tensor, 0)
    
    @property
    def right(self) -> Hand:
        return Hand(self.tensor, Hand.TSIZE)

@dataclass
class Gesture:
    MAX_FRAMES = 64
    
    tensor: torch.Tensor

    def __init__(self, tensor=None):
        self.tensor = tensor
        if tensor is None:
            self.reset()

    def reset(self):
        self.tensor = torch.zeros((self.MAX_FRAMES, GestureFrame.TSIZE))
    
    def capture(self, detection):
        frame = torch.zeros((1,GestureFrame.TSIZE))

        for i in range(len(detection.multi_handedness)):
            index = detection.multi_handedness[i].classification[0].index

            for i, landmark in enumerate(detection.multi_hand_landmarks[i].landmark):
                frame[0,index*Hand.TSIZE+3*i] = landmark.x
                frame[0,index*Hand.TSIZE+3*i+1] = landmark.y
                frame[0,index*Hand.TSIZE+3*i+2] = landmark.z
        
        self.tensor = torch.cat([self.tensor[1:], frame], dim=0)

    def save(self, filename):
        torch.save(self.tensor, filename)

    @classmethod
    def load(cls, filename) -> "Gesture":
        return cls(torch.load(filename))
        
class GestureVarients:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

        self.path.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(list(self.path.glob('*.gesture')))

    def save(self, gesture: Gesture):
        gesture.save(self.path/f"{len(self)}.gesture")

    def __getitem__(self, index) -> Gesture:
        return Gesture.load(self.get_file(index))
    
    def get_file(self, index) -> Path:
        return self.path/f"{index}.gesture"
    
class GestureLibrary:
    def __init__(self, root: Union[Path, str]):
        self.root = root

        if isinstance(self.root, str):
            self.root = Path(self.root)

    def get(self, name) -> GestureVarients:
        if name not in self.keys():
            raise KeyError(f"Undefined gesture {name}")

        return GestureVarients(name, self.root / name)
    
    @classmethod
    def keys(cls) -> List[str]:
        # this is a classmethod so that gesture class index is consistant
        return [
            'right_thumb_up', 
            'left_thumb_up', 
            'right_thumb_down', 
            'left_thumb_down', 
            'right_one', 
            'left_one', 
            'right_ok', 
            'left_ok',
            'ten',
        ]