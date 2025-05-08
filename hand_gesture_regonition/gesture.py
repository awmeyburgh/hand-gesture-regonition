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
    TSIZE = 3
    tensor: torch.Tensor
    offset: int
    
    @property
    def x(self):
        return self.tensor[0, self.offset].item()
    
    @property
    def y(self):
        return self.tensor[1, self.offset].item()
    
    @property
    def z(self):
        return self.tensor[2, self.offset].item()

@dataclass
class Hand:
    TSIZE = 21
    
    tensor: torch.Tensor
    
    @property
    def landmarks(self) -> List[Landmark]:
        return [Landmark(self.tensor, i) for i in range(Hand.TSIZE)]
            

@dataclass
class GestureFrame:
    TSIZE = Hand.TSIZE + 1
    
    tensor: torch.Tensor
    
    @property
    def hand(self) -> Hand:
        return Hand(self.tensor)
    
    @property
    def is_right(self) -> bool:
        return self.tensor[GestureFrame.TSIZE-1, 0].item() == 1

@dataclass
class Gesture:
    MAX_FRAMES = 64
    
    tensor: torch.Tensor

    def __init__(self, tensor=None, is_right=None):
        if tensor is None:
            self.reset()
            if is_right is None:
                raise Exception("is_right must be specified")
            self.is_right = is_right
        else:
            self.tensor = tensor
            self.is_right = (self.tensor[0, 0, GestureFrame.TSIZE-1] == 1).item()

    def reset(self):
        self.tensor = torch.zeros((self.MAX_FRAMES, Landmark.TSIZE, GestureFrame.TSIZE))
    
    def capture(self, detection) -> bool:
        if not detection.multi_handedness:
            return False
        
        frame = None

        for i in range(len(detection.multi_handedness)):
            is_right = detection.multi_handedness[i].classification[0].index == 1
            
            if is_right == self.is_right:
                frame = torch.zeros((1,Landmark.TSIZE,GestureFrame.TSIZE))
                
                for i, landmark in enumerate(detection.multi_hand_landmarks[i].landmark):
                    frame[0,0, i] = landmark.x
                    frame[0,1, i] = landmark.y
                    frame[0,2, i] = landmark.z
                    
                frame[0, :, GestureFrame.TSIZE-1] = torch.Tensor([is_right * 1]).repeat(3)
                
        if frame is None:
            return False
        
        self.tensor = torch.cat([self.tensor[1:], frame], dim=0)
        
        return True

    def save(self, filename):
        torch.save(self.tensor, filename)

    @classmethod
    def load(cls, filename) -> "Gesture":
        return cls(torch.load(filename))
    
    @property
    def is_static(self) -> bool:
        return torch.all(self.tensor[:-1] == 0).item()
        
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
    
    @property
    def is_static(self):
        return self.name.startswith('s_')
    
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
            's_right_thumb_up', 
            's_left_thumb_up', 
            's_right_thumb_down', 
            's_left_thumb_down', 
            's_right_index', 
            's_left_index', 
            's_right_ok', 
            's_left_ok',
            's_right_hand', 
            's_left_hand',
            's_right_peace', 
            's_left_peace',
        ]
        
def migration():
    library = GestureLibrary('data')
    for key in library.keys():
        from_variants = library.get(key)
        to_key = '_' + key
        to_variants = GestureVarients(to_key, library.root / to_key)
        for i in range(len(from_variants)):
            from_tensor = torch.load(from_variants.get_file(i))
            to_tensor = torch.transpose(from_tensor, 1, 2)
            torch.save(to_tensor, to_variants.get_file(i))