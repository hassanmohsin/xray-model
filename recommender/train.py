import json
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from recommender.dataset import AgentDataset
from recommender.models import BaselineModel

writer = SummaryWriter("./test")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def train_agent(agent_name, evaluate_only=False):
    batch_size = 256
    epochs = 5
    learning_rate = 0.0001
    num_workers = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # TODO: Use the same normalization values as the original ones
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = AgentDataset(
        f"/data2/mhassan/xray-model/performances/{agent_name}-performance-validation-set.csv",
        img_dir="/data2/mhassan/xray-dataset/v3/validation-set",
        transform=transform
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    validation_set = AgentDataset(
        f"/data2/mhassan/xray-model/performances/{agent_name}-performance-test-set.csv",
        img_dir="/data2/mhassan/xray-dataset/v3/test-set",
        transform=transform
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
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
        checkpoint = torch.load(f"./recomm_models/baseline/{agent_name}-checkpoint-best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        indices = []
        predictions = []
        with torch.no_grad():
            for idx, image, label in tqdm(validation_loader, desc=f"Evaluating {agent_name}"):
                inputs = image.to(device)
                preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                predictions += preds.tolist()
                indices += idx

        pd.DataFrame(
            {
                'ImageId': indices,
                'Label': predictions
            }
        ).to_csv(
            f'{agent_name}-prediction-probability.csv',
            index=False
        )

        return

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        for i, (idx, x, y) in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = x.to(device), y.to(device).unsqueeze(1)
            labels = labels.to(torch.float)  # For BCEwithLogits loss

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

        train_loss = epoch_loss / len(data_loader)
        train_acc = epoch_acc / len(data_loader)
        print(f'Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc: {train_acc:.3f}')
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
        is_best = bool(train_acc > best_acc)
        best_acc = max(best_acc, train_acc)
        save_checkpoint(
            {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'loss': train_loss,
                'best_acc': best_acc
            },
            is_best,
            filename=f'./recomm_models/baseline/{agent_name}-checkpoint-best.pth.tar'
        )

        # add to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
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

def evaluate(agents, assignment, assignment_type="optimized"):
    # Evaluate the result
    ind_eval = {}
    evaluation = {}
    total_acc, total_fbeta = 0., 0.
    for agent in agents:
        # Read the test performance file
        performance_df = pd.read_csv(f"./performances/{agent}-performance-test-set.csv", dtype={"image_id": str})
        # individual performance
        acc = performance_df.performance.sum() / len(performance_df)
        fbeta = metrics.fbeta_score(
            performance_df.label.to_list(),
            performance_df.prediction.to_list(),
            beta=2.0
        )
        ind_eval[agent] = {
            "accuracy": f"{acc:.3f}",
            "fbeta": f"{fbeta:.3f}"
        }
        # assigned performance
        _df = performance_df[performance_df.image_id.isin(assignment[agent])]
        acc = _df.performance.sum() / len(_df)
        total_acc += acc
        fbeta = metrics.fbeta_score(_df.label.to_list(), _df.prediction.to_list(), beta=2.0)
        total_fbeta += fbeta
        evaluation[agent] = {
            "accuracy": f"{acc:.3f}",
            "fbeta": f"{fbeta:.3f}"
        }

    evaluation['avg_acc'] = f"{total_acc / len(agents):.3f}"
    evaluation['avg_fbeta'] = f"{total_fbeta / len(agents):.3f}"
    final_eval = {
        'individual': ind_eval,
        'assigned': evaluation
    }
    print(final_eval)
    with open(f"./recommender_evaluation-{assignment_type}.json", 'w') as f:
        json.dump(final_eval, f)


def main(train=True):
    agents = [
        "agent_one",
        "agent_two",
        "agent_three",
        "agent_four",
        "agent_five",
        "agent_six",
        "agent_seven",
        "agent_eight"
    ]

    if train:
        # # Train and evaluate to get the probabilities
        for agent in agents:
            print(f"Training and evaluating {agent}")
            train_agent(agent_name=agent)
            train_agent(agent_name=agent, evaluate_only=True)

    probabilities = pd.concat(
        [
            pd.read_csv(
                f"{agent_name}-prediction-probability.csv",
                dtype={'ImageId': str, 'Label': float}
            ).set_index('ImageId') for agent_name in agents
        ],
        axis=1
    ).transpose()

    # Find optimized assignment
    assignment = []
    random_assignment = []
    for i in range(0, probabilities.shape[1], len(agents)):
        df = probabilities.iloc[:, i:i + len(agents)]
        img_ids = df.columns
        _agent_indices = linear_sum_assignment(df.to_numpy(), maximize=True)[1]
        assignment.append(img_ids[_agent_indices].tolist())
        img_ids = img_ids.tolist()
        np.random.shuffle(img_ids)
        random_assignment.append(img_ids)

    # TODO: Take care of the last element that has less than agent_count elements
    assignment = pd.DataFrame.from_records(
        np.stack(assignment[:-1], axis=0),
        columns=agents
    )
    assignment.to_csv(
        "assignment.csv",
        index=False
    )
    random_assignment = pd.DataFrame.from_records(
        np.stack(random_assignment[:-1], axis=0),
        columns=agents
    )

    print("Evaluating optimized assignment")
    evaluate(agents, assignment, assignment_type="optimized")
    print("Evaluating random assignment")
    evaluate(agents, random_assignment, assignment_type="random")


if __name__ == '__main__':
    main(train=False)
