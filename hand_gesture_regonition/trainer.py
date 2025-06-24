from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from hand_gesture_regonition import dataset
import torch
from torch.utils.data import DataLoader

from hand_gesture_regonition.gesture import migration
from hand_gesture_regonition.network import StaticGRNetwork
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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

    ALL_Y = np.concatenate(ALL_Y)
    ALL_Y_pred = np.concatenate(ALL_Y_pred)
    b = np.zeros_like(ALL_Y_pred)
    b[np.arange(len(ALL_Y_pred)), ALL_Y_pred.argmax(1)] = 1
    ALL_Y_pred = b

    print(f'Accuracy: {accuracy_score(ALL_Y, ALL_Y_pred, normalize=True)}')
    print(f'Recall: {recall_score(ALL_Y, ALL_Y_pred, average='weighted')}')
    print(f'Precision: {precision_score(ALL_Y, ALL_Y_pred, average='weighted')}')
    print(f'F1: {f1_score(ALL_Y, ALL_Y_pred, average='weighted')}')

    # print(f'Accuracy: {correct}/{total} ({correct/total:.2%})')

def train_random_forest(trainloader: DataLoader) -> RandomForestClassifier:
    ALL_X = []
    ALL_Y = []

    for X_batch, Y_batch in trainloader:
        ALL_X.append(X_batch)
        ALL_Y.append(Y_batch)

    X_train = np.concatenate(ALL_X)
    y_train = np.concatenate(ALL_Y)

    X_train = X_train.reshape((X_train.shape[0], -1))



    estimator = RandomForestClassifier()
    search = GridSearchCV(
        estimator,
        {
            'n_estimators': [n for n in range(10, 100, 5)],
        },
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    return search.best_estimator_

def test_random_forest(estimator: RandomForestClassifier, testloader: DataLoader):
    ALL_X = []
    ALL_Y = []

    for X_batch, Y_batch in testloader:
        ALL_X.append(X_batch)
        ALL_Y.append(Y_batch)

    X_test = np.concatenate(ALL_X)
    y_test = np.concatenate(ALL_Y)

    X_test = X_test.reshape((X_test.shape[0], -1))

    y_pred = estimator.predict(X_test)

    print(f'RF Accuracy: {accuracy_score(y_test, y_pred, normalize=True)}')
    print(f'RF Recall: {recall_score(y_test, y_pred, average='weighted')}')
    print(f'RF Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0)}')
    print(f'RF F1: {f1_score(y_test, y_pred, average='weighted')}')


def static_main():
    do_load = True
    do_save = True
    do_train = False
    do_test = True
    
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
    test_random_forest(random_forest, testloader)
        
def main():
    # migration()
    do_static = True
    
    if do_static:
        static_main()