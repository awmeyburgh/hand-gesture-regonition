from email.policy import default
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import yaml
from hand_gesture_regonition.v2.dataset import load_dataset
from hand_gesture_regonition.v2.model.config import Config
from hand_gesture_regonition.v2.model.static.logistic_regression import StaticLogisticRegression
import typer
from typing import List, Optional, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

from hand_gesture_regonition.v2.model.static.random_forest import StaticRandomForest
from hand_gesture_regonition.v2.model.static.support_vector_machine import StaticSupportVectorMachine


def metrics(y_true, y_pred, labels):
    # y_true is now a 1D array of integer labels
    # y_pred is a probability distribution, so we need to get the class with the highest probability
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_labels, labels=range(len(labels)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "precision": precision_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist()
    }

    return metrics


def get_model(key: str, load: bool, **kwargs):
    config = Config.get().models[key]
    model = None

    match key:
        case "static/logistic_regression":
            model = StaticLogisticRegression(config, **kwargs)
        case "static/random_forest":
            model = StaticRandomForest(config, **kwargs)
        case "static/support_vector_machine":
            model = StaticSupportVectorMachine(config, **kwargs)

    if model is None:
        raise Exception(f"Unable to resolve model `{key}`")

    if load:
        model = model.load()

    return model

def parse_kwargs(params):
    kwargs = {}
    if params:
        for i in range(0, len(params), 2):
            if i + 1 < len(params):
                key_param = params[i].lstrip('-')
                value = params[i+1]
                if value == 'null':
                    kwargs[key_param] = None
                elif value.lower() == 'true':
                    kwargs[key_param] = True
                elif value.lower() == 'false':
                    kwargs[key_param] = False
                else:
                    value = value.strip('"') # Strip quotes if present
                    try:
                        # Attempt to convert to int, then float, otherwise keep as string
                        if '.' in value:
                            kwargs[key_param] = float(value)
                        else:
                            kwargs[key_param] = int(value)
                    except ValueError:
                        kwargs[key_param] = value
    return kwargs


def main(
    key: str = typer.Argument(..., help="Key for the model to run"),
    train_size: Optional[float] = typer.Option(0.7, help="Size of the training dataset (0.0-1.0)"),
    train: bool = typer.Option(False, "--train", help="Run training"),
    test: bool = typer.Option(False, "--test", help="Run testing"),
    save_model: bool = typer.Option(False, "--save", help="Save the trained model"),
    load_model: bool = typer.Option(False, "--load", help="Load a pre-trained model"),
    params: Optional[List[str]] = typer.Argument(None, help="Arbitrary key-value pairs, e.g., param1 value1 param2 value2"),
):
    kwargs = parse_kwargs(params)
    model = get_model(key, load_model, **kwargs)
    X_train, X_test, y_train, y_test, labels = load_dataset(key[:key.index('/')], train_size=train_size)

    results = {}

    if train:
        y_pred_train = model.train(X_train, y_train)
        results['train'] = metrics(y_train, y_pred_train, labels=labels)

    if test:
        y_pred_test = model.predict(X_test)
        results['test'] = metrics(y_test, y_pred_test, labels=labels)

    with open('data/v2/model/results.yaml', 'w') as f:
        yaml.dump(results, f)

    if save_model:
        model.save()

if __name__ == "__main__":
    typer.run(main)