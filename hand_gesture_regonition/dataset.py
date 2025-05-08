import math
from pathlib import Path
from typing import List, Tuple

import torch
from hand_gesture_regonition.gesture import Gesture, GestureLibrary, Hand
from torch.utils.data import Dataset

from hand_gesture_regonition.network import StaticGRNetwork

class GestureLibraryDataset(Dataset):
    def __init__(self, library: GestureLibrary, train=True, train_subset=0.8,static=True):
        super().__init__()

        self.library = library
        self.train = train
        self.static = static

        self.gesture_variant_files = self.collect_gesture_variant_files(train, train_subset)

    def collect_gesture_variant_files(self, train, train_subset) -> List[Tuple[int, Path]]:
        result = []

        for j, key in enumerate(self.library.keys()):
            if self.static and not key.startswith('s_'):
                continue
            if not self.static and key.startswith('s_'):
                continue
            
            variants = self.library.get(key)
            size = len(variants)
            train_size = math.floor(size * train_subset)

            index_start = 0 if train else train_size
            index_end = train_size if train else size

            k = None
            if self.static:
                k = StaticGRNetwork.classes().index(GestureLibrary.keys()[j])

            for i in range(index_start, index_end):
                result.append((k, variants.get_file(i)))

        return result

    def __len__(self):
        return len(self.gesture_variant_files)
    
    def __getitem__(self, index):
        cls, path = self.gesture_variant_files[index]

        x = Gesture.load(path).tensor
        
        y = torch.zeros(len(StaticGRNetwork.classes()))
        y[cls] = 1

        return x, y
    
def load(train_subset=0.8,static=True) -> Tuple[Dataset, Dataset]:
    library = GestureLibrary('data')
    
    return (
        GestureLibraryDataset(library, train=True, train_subset=train_subset,static=static),
        GestureLibraryDataset(library, train=False, train_subset=train_subset,static=static),
    )