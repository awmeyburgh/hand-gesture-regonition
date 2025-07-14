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

app = typer.Typer()

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

    with open(Path('data/v2/model/results.yaml'), 'w') as f:
        yaml.dump(metrics, f)


def get_model(key: str, load: bool, **kwargs):
    config = Config.get().models[key]
    model = None

    match key:
        case "static/logistic_regression":
            model = StaticLogisticRegression(config, **kwargs)

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
                kwargs[key_param] = value
    return kwargs


@app.command()
def train(
    key: str = typer.Argument(..., help="Key for the model to train"),
    save: Optional[bool] = typer.Option(False, help="Save the trained model"),
    load: Optional[bool] = typer.Option(False, help="Load a pre-trained model"),
    train_size: Optional[float] = typer.Option(0.7, help="Size of the training dataset (0.0-1.0)"),
    params: Optional[List[str]] = typer.Argument(None, help="Arbitrary key-value pairs, e.g., --param1 value1 --param2 value2"),
):
    kwargs = parse_kwargs(params)

    model = get_model(key, load, **kwargs)

    X, _, y, _, labels = load_dataset(key[:key.index('/')], train_size=train_size)

    y_pred = model.predict(X)
    metrics(y, y_pred, labels=labels)

    if save:
        model.save()

@app.command()
def test(
    key: str = typer.Argument(..., help="Key for the model to test"),
    save: Optional[bool] = typer.Option(False, help="Save test results"),
    load: Optional[bool] = typer.Option(False, help="Load a model for testing"),
    train_size: Optional[float] = typer.Option(0.7, help="Size of the training dataset (0.0-1.0)"),
    params: Optional[List[str]] = typer.Argument(None, help="Arbitrary key-value pairs, e.g., --param1 value1 --param2 value2"),
):
    kwargs = parse_kwargs(params)

    model = get_model(key, load, **kwargs)

    _, X, _, y, labels = load_dataset(key[:key.index('/')], train_size=train_size)

    y_pred = model.predict(X)
    metrics(y, y_pred, labels=labels)

    if save:
        model.save()

if __name__ == "__main__":
    app()