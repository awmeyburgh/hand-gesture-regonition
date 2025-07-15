from sklearn.ensemble import RandomForestClassifier
from hand_gesture_regonition.v2.model.sklearn_model import SklearnModel


class StaticRandomForest(SklearnModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def estimator(self):
        print(self.kwargs)
        return RandomForestClassifier(**self.kwargs, n_jobs=-1)
