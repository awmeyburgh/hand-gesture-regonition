from torch import nn
import torch

from hand_gesture_regonition.gesture import Gesture, GestureLibrary

class GestureRegonitionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.lstm = nn.ModuleList([
        #     nn.LSTM(GestureFrame.TSIZE, 30, Gesture.MAX_FRAMES, batch_first=True),
        # ])
        
        # self.output = nn.Sequential(
        #     nn.Linear(30, 30),
        #     nn.Sigmoid(),
        #     nn.Linear(30, len(GestureLibrary.keys())),
        #     nn.Sigmoid()
        # )
        
        # self.layers = nn.Sequential(
        #     nn.Conv1d(Gesture.MAX_FRAMES, 256, 3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2),
        #     nn.Conv1d(256, 1024, 11),
        #     nn.Sigmoid(),
        #     nn.MaxPool1d(2),
        #     nn.Flatten(),
        #     nn.Sigmoid(),
        #     nn.Linear(26624, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, len(GestureLibrary.keys())),
        #     nn.Sigmoid()
        # )
        
        self.layers = nn.Sequential(
            nn.Conv1d(Gesture.MAX_FRAMES, 1024, 3),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(7936, 100),
            nn.ReLU(),
            nn.Linear(100, len(GestureLibrary.keys())),
            nn.Sigmoid()
        )
        
        self.criterion = nn.BCELoss()
        
    def forward(self, X):
        # hc = [
        #     (
        #         torch.zeros((Gesture.MAX_FRAMES, X.shape[0], 30)),
        #         torch.zeros((Gesture.MAX_FRAMES, X.shape[0], 30))
        #     )
        # ]
        
        # for i in range(len(self.lstm)):
        #     X, _ = self.lstm[i](X, hc[i])
            
        # return self.output(X[:, -1, :])
        return self.layers(X)
        
    def backward(self, Y_pred, Y) -> float:
        loss = self.criterion(Y_pred, Y)
        loss.backward()
        
        return loss.item()
        
    def save(self, filename):
        torch.save(self.state_dict(), filename)
        
    @classmethod
    def load(cls, filename, error_ok=True) -> "GestureRegonitionNetwork":
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