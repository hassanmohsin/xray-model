import os

import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from recommender.dataset import AgentDataset
from recommender.models import BaselineModel

writer = SummaryWriter("./test")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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


def train(agent_name, evaluate_only=False):
    batch_size = 64
    epochs = 20
    learning_rate = 0.0001
    num_workers = 56
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = "../dummy-dataset/test-set"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = AgentDataset(
        "./probabilities.csv",
        agent_name,
        img_dir=dataset_dir,
        transform=transform
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    model = BaselineModel()
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if evaluate_only:
        print("Evaluating on the test set...")
        checkpoint = torch.load(f"./saved_recomm_models/baseline/{agent_name}-checkpoint.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        indices = []
        predictions = []
        with torch.no_grad():
            for i, (idx, image, label) in enumerate(data_loader, 0):
                inputs = image.to(device)
                preds = model(inputs).squeeze().cpu().numpy()
                predictions += preds.tolist()
                indices += idx.tolist()

        pd.DataFrame(
            {
                'ImageId': indices,
                'Label': predictions
            }
        ).to_csv(
            'recommender_preds.csv',
            index=False
        )

        return

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = 100
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        for i, (idx, x, y) in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = x.to(device), y.to(device).unsqueeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            running_loss += loss.item()
            if i % 10 == 0:
                print(f'iter: {i:04}: Running loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        train_loss = epoch_loss / len(data_loader)
        print(f'Epoch {epoch:03}: Loss: {train_loss:.3f}')
        # model.eval()
        # val_accuracy = 0.0
        # val_loss = 0.0
        # with torch.no_grad():
        #     for i, (x, y) in enumerate(validation_loader, 0):
        #         inputs, labels = x.to(device), y.to(device).unsqueeze(1)
        #         labels = labels.to(torch.float)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         acc = binary_acc(outputs, labels)
        #         val_accuracy += acc.item()
        #         val_loss += loss.item()
        #
        # acc = val_accuracy / len(validation_loader)
        # val_loss = val_loss / len(validation_loader)
        # is_best = bool(acc > best_accuracy)
        # best_accuracy = max(acc, best_accuracy)
        # Save checkpoint if is a new best
        is_best = train_loss < best_loss
        best_loss = min(best_loss, train_loss)
        save_checkpoint(
            {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'loss': train_loss,
                'best_loss': best_loss
            },
            is_best,
            filename=f'./saved_recomm_models/baseline/{agent_name}-checkpoint.pth.tar'
        )

        # add to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("Loss/validation", val_loss, epoch)

        writer.flush()

    print('Finished Training.')


# def main():
#     parser = ArgumentParser(description='Train a model in xray images')
#     parser.add_argument('--input', type=str, required=True, action='store',
#                         help="JSON input")
#     args = parser.parse_args()
#     if not os.path.isfile(args.input):
#         raise FileNotFoundError("Input {args.input} not found.")
#
#     with open(args.input) as f:
#         args = json.load(f)
#
#     if not os.path.isdir(args['model_dir']):
#         os.makedirs(args['model_dir'])
#
#     # train the model
#     train(args, evaluate_only=False)
#
#     # generate the submission file
#     train(args, evaluate_only=True)


if __name__ == '__main__':
    train(agent_name="agent_one")
    train(agent_name="agent_two")
    train(agent_name="agent_three")