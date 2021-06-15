import json
import multiprocessing as mp
import os
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import XrayImageDataset
from .models import BaselineModel, ModelOne, ModelTwo

writer = SummaryWriter("./test")

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


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


def train(args, evaluate_only=True):
    dataset_dir = args['dataset_dir']
    train_dir = os.path.join(dataset_dir, 'train-set')
    valid_dir = os.path.join(dataset_dir, 'validation-set')
    test_dir = os.path.join(dataset_dir, 'test-set')

    batch_size = args['batch_size']
    epochs = args['epochs']
    learning_rate = args['learning_rate']
    num_workers = args['num_worker']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_data = XrayImageDataset(os.path.join(dataset_dir, 'train-labels-short.csv'), train_dir)
    validation_data = XrayImageDataset(os.path.join(dataset_dir, 'validation-labels-short.csv'), valid_dir)
    test_data = XrayImageDataset(os.path.join(dataset_dir, 'sample-submission.csv'), test_dir, True)

    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    model = BaselineModel()
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    if evaluate_only:
        print("Evaluating on the test set...")
        checkpoint = torch.load(os.path.join(args['model_dir'], "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        indices = []
        predictions = []
        with torch.no_grad():
            for i, (idx, x) in enumerate(test_loader, 0):
                inputs = x.to(device)
                outputs = model(inputs)
                preds = torch.round(torch.sigmoid(outputs)).squeeze().to(torch.int).cpu().numpy()
                indices += idx
                predictions += preds.tolist()

        df = pd.DataFrame({'ImageId': indices, 'Label': predictions})
        df.to_csv(os.path.join(args['model_dir'], 'submission.csv'), index=False)

        return

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
        save_checkpoint(
            {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'loss': val_loss,
                'best_accuracy': best_accuracy
            },
            is_best,
            filename=os.path.join(args['model_dir'], 'checkpoint.pth.tar')
        )

        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_acc / len(train_loader)
        print(
            f'Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc:'
            f' {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {best_accuracy:.3f}'
        )

        # add to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Acc/validation", val_accuracy, epoch)

        writer.flush()

    print('Finished Training.')


def main():
    parser = ArgumentParser(description='Train a model in xray images')
    parser.add_argument('--input', type=str, required=True, action='store',
                        help="JSON input")
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError("Input {args.input} not found.")

    with open(args.input) as f:
        args = json.load(f)

    if not os.path.isdir(args['model_dir']):
        os.makedirs(args['model_dir'])

    # train the model
    train(args, evaluate_only=False)

    # generate the submission file
    train(args, evaluate_only=True)


if __name__ == '__main__':
    main()
