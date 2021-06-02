import multiprocessing as mp
import os

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from .baseline import BaselineModel
from .dataset import XrayImageDataset


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def save_checkpoint(state, is_best, filename='./output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


def train():
    dataset_dir = '/home/mhassan/xray/dataset'
    train_dir = os.path.join(dataset_dir, 'train-set')
    valid_dir = os.path.join(dataset_dir, 'validation-set')
    test_dir = os.path.join(dataset_dir, 'test-set')

    batch_size = 256
    epochs = 10
    learning_rate = 0.001
    momentum = 0.9
    num_workers = mp.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    training_data = XrayImageDataset(os.path.join(dataset_dir, 'train-labels.csv'), train_dir, transform)
    validation_data = XrayImageDataset(os.path.join(dataset_dir, 'validation-labels.csv'), valid_dir, transform)

    train_loader = DataLoader(training_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)

    model = BaselineModel()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = x.to(device), y.to(device).unsqueeze(1)
            labels = labels.to(torch.float)  # for BCEwithLogits loss

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = binary_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            running_loss += loss.item()
            running_acc += acc.item()
            if i % 10 == 0:
                print(f'iter: {i:04}: Running loss: {running_loss / 10:.3f} | Running acc: {running_acc / 10:.3f}')
                running_loss = 0.0
                running_acc = 0.0

        model.eval()
        val_accuracy = 0.0
        val_loss = 0.0
        best_accuracy = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(validation_loader, 0):
                inputs, labels = x.to(device), y.to(device).unsqueeze(1)
                labels = labels.to(torch.float)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = binary_acc(outputs, labels)
                val_accuracy += acc.item()
                val_loss += loss.item()

        acc = val_accuracy / len(validation_loader)
        val_loss = val_loss / len(validation_loader)
        is_best = bool(acc > best_accuracy)
        best_accuracy = max(acc, best_accuracy)
        # Save checkpoint if is a new best
        save_checkpoint({
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'loss': val_loss,
            'best_accuracy': best_accuracy
        }, is_best, filename=f"./output/checkpoint.pth.tar")

        print(f'Epoch {epoch:03}: Loss: {epoch_loss / len(train_loader):.3f} | Acc:'
              f' {epoch_acc / len(train_loader):.3f} | Val Loss: {val_loss:.3f} | Val Acc: {best_accuracy:.3f}')

    print('Finished Training')


if __name__ == '__main__':

    # For saving the models
    if not os.path.isdir('./output'):
        os.makedirs('./output')

    train()
