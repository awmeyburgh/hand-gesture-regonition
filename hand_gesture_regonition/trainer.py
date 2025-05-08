from pathlib import Path
from hand_gesture_regonition import dataset
import torch
from torch.utils.data import DataLoader

from hand_gesture_regonition.gesture import migration
from hand_gesture_regonition.network import StaticGRNetwork

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
    with torch.no_grad():
        for (X, Y) in testloader:
            Y_pred = modal.forward(X)
        
            total += Y.size(0)
            correct += modal.eval_y(Y_pred, Y).sum().item()

    print(f'Accuracy: {correct}/{total} ({correct/total:.2%})')

def static_main():
    do_load = True
    do_save = True
    do_train = True
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
        
def main():
    # migration()
    do_static = True
    
    if do_static:
        static_main()