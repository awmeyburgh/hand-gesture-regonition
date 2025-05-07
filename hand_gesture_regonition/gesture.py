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
    x: double
    y: double
    z: double

    @classmethod
    def parse(cls, landmark) -> "Landmark":
        return cls(landmark.x, landmark.y, landmark.z)
    
    def __iter__(self):
        return iter([self.x, self.y, self.z])

@dataclass
class Hand:
    landmarks: List[Landmark]

    @classmethod
    def parse(cls, landmarks) -> "Hand":
        return cls([
            Landmark.parse(landmark)
            for landmark in landmarks
        ])
    
    def to_tensor(self):
        landmarks = []
        for landmark in self.landmarks:
            landmarks.extend(list(landmark))

        return torch.Tensor(landmarks)

@dataclass
class GestureFrame:
    left: Hand
    right: Hand
        
    def to_tensor(self) -> torch.Tensor:
        result = torch.zeros(2*21*3)

        if self.left is not None:
            result[:21*3] = self.left.to_tensor()
        
        if self.right is not None:
            result[21*3:] = self.right.to_tensor()

        return result

@dataclass
class Gesture:
    frames: List[GestureFrame]

    def __init__(self):
        self.frames = []

    def capture(self, detection):
        hands = [None, None]

        for i in range(len(detection.multi_handedness)):
            index = detection.multi_handedness[i].classification[0].index

            hands[index] = Hand.parse(detection.multi_hand_landmarks[i].landmark)
        
        self.frames.append(GestureFrame(*hands))

    @classmethod
    def decode(cls, frames) -> "Gesture":
        gesture = cls()

        for frame in frames['frames']:
            hands = [None, None]

            for hand_key, hand in frame.items():
                if hand is not None:
                    landmarks = []

                    for landmark in hand['landmarks']:
                        landmarks.append(Landmark(landmark['x'],landmark['y'],landmark['z']))

                    index = (hand_key == 'right') * 1

                    hands[index] = Hand(landmarks)
            
            gesture.frames.append(GestureFrame(*hands))
        
        return gesture


    def encode(self):
        return asdict(self)


    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.encode(), f)

    @classmethod
    def load(cls, filename) -> "Gesture":
        with open(filename, 'r') as f:
            return cls.decode(json.load(f))
        
    def to_tensor(self) -> torch.Tensor:
        result = torch.zeros((len(self.frames), 2*21*3))

        for i, frame in enumerate(self.frames):
            result[i, :] = frame.to_tensor()

        return result
        
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
        return ['left_wave', 'right_wave', 'left_move_up', 'right_move_up']
    
class GestureLibraryDataset(Dataset):
    def __init__(self, library: GestureLibrary, train=True, train_subset=0.8):
        super().__init__()

        self.library = library

        self.gesture_variant_files = self.collect_gesture_variant_files(train, train_subset)

    def collect_gesture_variant_files(self, train, train_subset) -> List[Tuple[int, Path]]:
        result = []

        for j, key in enumerate(self.library.keys()):
            variants = self.library.get(key)
            size = len(variants)
            train_size = math.floor(size * train_subset)

            index_start = 0 if train else train_size
            index_end = train_size if train else size

            for i in range(index_start, index_end):
                result.append((j, variants.get_file(i)))

        return result

    def __len__(self):
        return len(self.gesture_variant_files)
    
    def __getitem__(self, index):
        cls, path = self.gesture_variant_files[index]

        return Gesture.load(path).to_tensor(), cls