from sklearn.linear_model import LogisticRegression
from hand_gesture_regonition.v2.model.sklearn_model import SklearnModel


class StaticLogisticRegression(SklearnModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def estimator(self):
        return LogisticRegression(**self.kwargs, n_jobs=-1)