from abc import abstractmethod
import pickle


from hand_gesture_regonition.v2.model.base import BaseModel
from hand_gesture_regonition.v2.model.config import ModelConfig


class SklearnModel(BaseModel):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.__estimator = self.estimator()

    @abstractmethod
    def estimator(self):
        pass

    def train(self, X, y):
        self.__estimator = self.__estimator.fit(X, y)
        return self.__estimator.predict_proba(X)
    
    def predict(self, X):
        return self.__estimator.predict_proba(X)

    def load(self):
        try:
            with open(self.path, 'rb') as f:
                print(f"Loaded `{self.config.key}`")
                return pickle.load(f)
        except Exception as e:
            print(f"Exception Loading `{self.config.key}`: {e}")
            return self

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved `{self.config.key}`")

    