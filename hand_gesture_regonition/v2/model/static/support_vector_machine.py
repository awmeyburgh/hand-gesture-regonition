from hand_gesture_regonition.v2.model.base import BaseModel
from sklearn.svm import SVC

from hand_gesture_regonition.v2.model.sklearn_model import SklearnModel


class StaticSupportVectorMachine(SklearnModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def estimator(self):
        return SVC(**self.kwargs, probability=True)