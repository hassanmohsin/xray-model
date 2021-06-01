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


if __name__ == '__main__':
    dataset_dir = '/home/mhassan/xray/dataset'
    train_dir = os.path.join(dataset_dir, 'train-set')
    valid_dir = os.path.join(dataset_dir, 'validation-set')
    test_dir = os.path.join(dataset_dir, 'test-set')

    batch_size = 32
    epochs = 2
    learning_rate = 0.001
    momentum = 0.9
    num_workers = mp.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    training_data = XrayImageDataset(os.path.join(dataset_dir, 'train-labels.csv'), train_dir)
    validation_data = XrayImageDataset(os.path.join(dataset_dir, 'validation-labels.csv'), valid_dir)

    train_loader = DataLoader(training_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=False)

    model = BaselineModel()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        epoch_loss = 0.0
        epoch_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = x.to(device), y.to(device).unsqueeze(1)

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
                print(f'iter: {i}, Running loss: {running_loss / 10:.5f}')
                running_loss = 0.0
                running_acc = 0.0

        print(f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc:'
              f' {epoch_acc / len(train_loader):.3f}')

    print('Finished Training')
