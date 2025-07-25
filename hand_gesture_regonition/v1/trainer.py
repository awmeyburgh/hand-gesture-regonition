from pathlib import Path
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from hand_gesture_regonition.v1 import dataset
import torch
from torch.utils.data import DataLoader

from hand_gesture_regonition.v1.gesture import migration
from hand_gesture_regonition.v1.network import StaticGRNetwork
from sklearn.metrics import classification_report

def load_modal(do_load, path) -> StaticGRNetwork:
    if path.exists() and do_load:
        return StaticGRNetwork.load(path)
    return StaticGRNetwork()

def load_dataset(batch_size, static):
    trainset, testset = dataset.load(static=static)
    
    return (
        DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True
        ),
        DataLoader(
            testset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True
        )
    )
    
def train(modal: StaticGRNetwork, trainloader: DataLoader, optimizer: torch.optim.Optimizer, rolling_batch_count, batch_size, epochs, loss_threshold):
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        running_loss = 0
        for i, (X, Y) in enumerate(trainloader, 0):
            Y_pred = modal.forward(X)
            loss = modal.backward(Y_pred, Y)
            
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss*batch_size
            # if i % rolling_batch_count == rolling_batch_count-1:
            
        print(f'[{epoch + 1}, {len(trainloader.dataset):5d}] loss: {running_loss / len(trainloader.dataset):.3f}')
        
        if running_loss / len(trainloader.dataset) < loss_threshold:
            break
                
def test(modal: StaticGRNetwork, testloader: DataLoader):
    correct = 0
    total = 0
    
    ALL_Y = []
    ALL_Y_pred = []

    with torch.no_grad():
        for (X, Y) in testloader:
            ALL_Y.append(Y.numpy())

            Y_pred = modal.forward(X)
            ALL_Y_pred.append(Y_pred.numpy())
        
            total += Y.size(0)
            correct += modal.eval_y(Y_pred, Y).sum().item()

    y_test = np.concatenate(ALL_Y)
    y_pred = np.concatenate(ALL_Y_pred)
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    y_pred = b

    print(classification_report(y_test, y_pred))

    # print(f'Accuracy: {correct}/{total} ({correct/total:.2%})')

def train_estimator(estimator, trainloader: DataLoader):
    ALL_X = []
    ALL_Y = []

    for X_batch, Y_batch in trainloader:
        ALL_X.append(X_batch)
        ALL_Y.append(Y_batch)

    X_train = np.concatenate(ALL_X)
    y_train = np.concatenate(ALL_Y)

    X_train = X_train.reshape((X_train.shape[0], -1))

    return estimator.fit(X_train, y_train)

def train_random_forest(trainloader: DataLoader) -> RandomForestClassifier:
    estimator = RandomForestClassifier()
    search = GridSearchCV(
        estimator,
        {
            'n_estimators': [n for n in range(10, 100, 5)],
        },
        n_jobs=-1
    )
    return train_estimator(search, trainloader).best_estimator_

def train_gradient_boosting(trainloader: DataLoader) -> GradientBoostingClassifier:
    estimator = GradientBoostingClassifier()
    search = GridSearchCV(
        estimator,
        {'n_estimators': [15, 50, 100, 150],
              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
              'subsample': [1.0, 0.5],
              'max_features': [1, 2, 3, 4]},
        n_jobs=-1
    )
    return train_estimator(search, trainloader).best_estimator_

def test_estimator(estimator, testloader: DataLoader, name='Random Forest'):
    ALL_X = []
    ALL_Y = []

    for X_batch, Y_batch in testloader:
        ALL_X.append(X_batch)
        ALL_Y.append(Y_batch)

    X_test = np.concatenate(ALL_X)
    y_test = np.concatenate(ALL_Y)

    X_test = X_test.reshape((X_test.shape[0], -1))

    y_pred = estimator.predict(X_test)

    print(f'{name}:')
    print(classification_report(y_test, y_pred))

def train_voting(estimators: List, trainloader: DataLoader) -> VotingClassifier:
    return train_estimator(
        VotingClassifier(estimators),
        trainloader
    )


def static_main():
    do_load = True
    do_save = True
    do_train = False
    do_test = False
    
    epochs = 1000
    batch_size = 100
    rolling_size = 10
    loss_threshold = 0.005
    
    save_path = Path('static_network.bin')
    modal = load_modal(do_load=do_load,path=save_path)
    optimizer = torch.optim.Adam(modal.parameters(), lr=1e-3)
    
    trainloader, testloader = load_dataset(batch_size=batch_size, static=True)
    
    if do_train:
        try:
            train(
                modal=modal,
                trainloader=trainloader,
                optimizer=optimizer,
                rolling_batch_count=rolling_size/batch_size,
                epochs=epochs,
                batch_size=batch_size,
                loss_threshold=loss_threshold
            )
        finally:
            if do_save:
                modal.save(save_path)
        
    if do_test:
        test(
            modal=modal,
            testloader=testloader
        )

    random_forest = train_random_forest(trainloader)
    test_estimator(random_forest, testloader, name='Random Forest')
    gradient_boosting = train_gradient_boosting(trainloader)
    test_estimator(gradient_boosting, testloader, name='Gradient Boosting')
    voting = train_voting([('RF', random_forest), ('GB', gradient_boosting)], trainloader)
    test_estimator(voting, testloader, name='Voting')
        
def main():
    # migration()
    do_static = True
    
    if do_static:
        static_main()