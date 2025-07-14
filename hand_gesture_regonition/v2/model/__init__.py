from hand_gesture_regonition.v2.model.base import BaseModel
from hand_gesture_regonition.v2.model.config import Config
from hand_gesture_regonition.v2.model.static.logistic_regression import StaticLogisticRegression


def load_model(key) -> BaseModel:
    config = Config.get().models[key]

    match key:
        case 'static/logistic_regression':
            return StaticLogisticRegression(config).load()