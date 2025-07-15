from hand_gesture_regonition.v2.model.base import BaseModel
from hand_gesture_regonition.v2.model.config import Config
from hand_gesture_regonition.v2.model.static.logistic_regression import StaticLogisticRegression
from hand_gesture_regonition.v2.model.static.random_forest import StaticRandomForest
from hand_gesture_regonition.v2.model.static.support_vector_machine import StaticSupportVectorMachine


def load_model(key) -> BaseModel:
    config = Config.get().models[key]

    match key:
        case 'static/logistic_regression':
            return StaticLogisticRegression(config).load()
        case "static/random_forest":
            return StaticRandomForest(config).load()
        case "static/support_vector_machine":
            return StaticSupportVectorMachine(config).load()

    raise Exception(f"Unable to resolve model `{key}`")