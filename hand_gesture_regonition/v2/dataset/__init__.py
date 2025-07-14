from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from hand_gesture_regonition.v2.dataset import sequential, static

def load_dataset(type: str, train_size: float = 0.7, random_state=0):
    if type == 'static':
        df = static.load()
    else:
        df = sequential.load()

    X = df.drop(["name", "gesture"], axis=1).to_numpy()
    y = df["gesture"]

    # Convert string labels to integer labels and get unique labels
    y_int_labels, unique_labels = pd.factorize(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int_labels,
        train_size=train_size,
        random_state=random_state,
        stratify=y_int_labels
    )

    return X_train, X_test, y_train, y_test, unique_labels