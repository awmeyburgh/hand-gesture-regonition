import math
from pathlib import Path
from typing import List, Tuple

import torch
from hand_gesture_regonition.gesture import Gesture, GestureLibrary, Hand
from torch.utils.data import Dataset

class GestureLibraryDataset(Dataset):
    def __init__(self, library: GestureLibrary, train=True, train_subset=0.8):
        super().__init__()

        self.library = library
        self.train = train

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

        x = Gesture.load(path).tensor
        
        if self.train:
            padding = 0
            for i in range(Gesture.MAX_FRAMES):
                if torch.all(x[i] == 0):
                    padding += 1
                else:
                    break
            
            if padding > 0:
                index = torch.randint(0, padding+1, (1,)).item()
                size = Gesture.MAX_FRAMES - padding
                
                _x = torch.randn(x.shape)/100
                _x[index:index+size] = x[padding:]
                
                x = _x
                
            # key = self.library.keys()[cls]
            # if key.startswith('right_'):
            #     x[:, :Hand.TSIZE] = torch.randn(Gesture.MAX_FRAMES, Hand.TSIZE)/10
            # if key.startswith('left_'):
            #     x[:, Hand.TSIZE:] = torch.randn(Gesture.MAX_FRAMES, Hand.TSIZE)/10
        
        y = torch.zeros(len(GestureLibrary.keys()))
        y[cls] = 1

        return x, y
    
def load(train_subset=0.8) -> Tuple[Dataset, Dataset]:
    library = GestureLibrary('data')
    
    return (
        GestureLibraryDataset(library, train=True, train_subset=train_subset),
        GestureLibraryDataset(library, train=False, train_subset=train_subset),
    )