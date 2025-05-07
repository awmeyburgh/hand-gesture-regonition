from pathlib import Path
from hand_gesture_regonition import dataset
import torch
from torch.utils.data import DataLoader

from hand_gesture_regonition.network import GestureRegonitionNetwork

def load_modal(do_load, path) -> GestureRegonitionNetwork:
    if path.exists() and do_load:
        return GestureRegonitionNetwork.load(path)
    return GestureRegonitionNetwork()

def load_dataset(batch_size):
    trainset, testset = dataset.load()
    
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
    
def train(modal: GestureRegonitionNetwork, trainloader: DataLoader, optimizer: torch.optim.Optimizer, rolling_batch_count, batch_size, epochs):
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
                
def test(modal: GestureRegonitionNetwork, testloader: DataLoader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (X, Y) in testloader:
            Y_pred = modal.forward(X)
        
            total += Y.size(0)
            correct += modal.eval_y(Y_pred, Y).sum().item()

    print(f'Accuracy: {correct}/{total} ({correct/total:.2%})')

def main():
    do_load = False
    do_save = True
    do_train = True
    do_test = True
    
    epochs = 1000
    batch_size = 40
    rolling_size = 10
    
    save_path = Path('network.bin')
    modal = load_modal(do_load=do_load,path=save_path)
    optimizer = torch.optim.Adam(modal.parameters(), lr=1e-4)
    
    trainloader, testloader = load_dataset(batch_size=batch_size)
    
    if do_train:
        try:
            train(
                modal=modal,
                trainloader=trainloader,
                optimizer=optimizer,
                rolling_batch_count=rolling_size/batch_size,
                epochs=epochs,
                batch_size=batch_size
            )
        finally:
            if do_save:
                modal.save(save_path)
        
    if do_test:
        test(
            modal=modal,
            testloader=testloader
        )