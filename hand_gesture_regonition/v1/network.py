from typing import List, Optional
from torch import nn
import torch

from hand_gesture_regonition.v1.gesture import Gesture, GestureLibrary

class StaticGRNetwork(nn.Module):
    __classes = None
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(3, 64, 2),
            nn.ReLU(),
            nn.Conv1d(64, 256, 2),
            nn.MaxPool1d(4),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(1280, 100),
            nn.ReLU(),
            nn.Linear(100, len(self.classes())),
            nn.Sigmoid(),
        )
        
        self.criterion = nn.BCELoss()
        
    def forward(self, X):
        if len(X.shape) == 4:
            # only do static frame if gesture is full movable frame
            X = X[:, -1, :, :]
            
        X = X.reshape(-1, 3, 22)
        
        return self.layers(X)
    
    def classify(self, gesture: Gesture, confidence=0.8) -> Optional[str]:
        x = gesture.tensor[-1]
        y_pred = self.forward(x)
        y_pred = y_pred >= confidence
        
        if y_pred.sum().item() == 1:
            _, index = torch.max(y_pred, 1)
            key = self.classes()[index]
            
            if gesture.is_right and 'left' in key:
                return None
            
            if not gesture.is_right and 'left' not in key:
                return None
            
            return key
        
        return None
        
    def backward(self, Y_pred, Y) -> float:
        loss = self.criterion(Y_pred, Y)
        loss.backward()
        
        return loss.item()
        
    def save(self, filename):
        torch.save(self.state_dict(), filename)
        
    @classmethod
    def load(cls, filename, error_ok=True) -> "StaticGRNetwork":
        result = cls()
        
        try:
            result.load_state_dict(torch.load(filename))
        except Exception as e:
            if error_ok:
                return result
            raise e
        
        return result
    
    @classmethod
    def eval_y(self, Y_pred, Y, confidence=0.8):
        Y_pred=Y_pred>=confidence
        Y=Y>=confidence
        
        result = torch.zeros(Y.size(0))
        
        for i in range(Y.size(0)):
            result[i] = torch.all(Y_pred[i]==Y[i]) * 1
            
        return result
    
    @classmethod
    def classes(cls) -> List[str]:
        if cls.__classes is None:
            cls.__classes = [key for key in GestureLibrary.keys() if key.startswith('s_')]
        return cls.__classes
    
class MovingGRNetwork(nn.Module):
    __classes = None
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(3, 64, 2),
            nn.ReLU(),
            nn.Conv1d(64, 256, 2),
            nn.MaxPool1d(4),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(1280, 100),
            nn.ReLU(),
            nn.Linear(100, len(self.classes())),
            nn.Sigmoid(),
        )
        
        self.criterion = nn.BCELoss()
        
    def forward(self, X):
        return self.layers(X)
    
    def classify(self, gesture: Gesture, confidence=0.8) -> Optional[str]:
        x = gesture.tensor[-1]
        y_pred = self.forward(x)
        y_pred = y_pred >= confidence
        
        if y_pred.sum().item() == 1:
            _, index = torch.max(y_pred, 1)
            key = self.classes()[index]
            
            if gesture.is_right and 'left' in key:
                return None
            
            if not gesture.is_right and 'left' not in key:
                return None
            
            return key
        
        return None
        
    def backward(self, Y_pred, Y) -> float:
        loss = self.criterion(Y_pred, Y)
        loss.backward()
        
        return loss.item()
        
    def save(self, filename):
        torch.save(self.state_dict(), filename)
        
    @classmethod
    def load(cls, filename, error_ok=True) -> "StaticGRNetwork":
        result = cls()
        
        try:
            result.load_state_dict(torch.load(filename))
        except Exception as e:
            if error_ok:
                return result
            raise e
        
        return result
    
    @classmethod
    def eval_y(self, Y_pred, Y, confidence=0.8):
        Y_pred=Y_pred>=confidence
        Y=Y>=confidence
        
        result = torch.zeros(Y.size(0))
        
        for i in range(Y.size(0)):
            result[i] = torch.all(Y_pred[i]==Y[i]) * 1
            
        return result
    
    @classmethod
    def classes(cls) -> List[str]:
        if cls.__classes is None:
            cls.__classes = [key for key in GestureLibrary.keys() if key.startswith('s_')]
        return cls.__classes