from abc import ABC, abstractmethod
from pathlib import Path

from hand_gesture_regonition.v2.model.config import ModelConfig


class BaseModel(ABC):
    def __init__(self, config: ModelConfig, **kwargs):
        self.config = config

        self.kwargs = {
            name: kwargs.get(name, p.default)
            for name, p in self.config.parameters.items()
        }

    @abstractmethod
    def train(self, X, y):
        """This should return a prob distribution"""
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @property
    def path(self) -> Path:
        return Path('data/v2/model')/self.config.type.lower()/self.config.filename
