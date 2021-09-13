import json
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

from recommender.dataset import AgentDataset
from recommender.models import BaselineModel
from recommender.performance import apply_dropout
from xray.agent import AgentGroup
from xray.config import AgentConfig
from xray.models import BaselineModel

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

seed = 42
# torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def save_checkpoint(state, is_best, filename='./output/checkpoint.pth.tar'):
    weight_dir = os.path.dirname(filename)
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, os.path.join(weight_dir, "checkpoint-best.pth.tar"))  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")
    torch.save(state, filename)


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for _, data, _ in loader:
        channels_sum += torch.mean(data, dim=(0, 2, 3))
        channels_squared_sum += torch.mean(data ** 2, dim=(0, 2, 3))
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# TODO: Remove `args` argument from the following method, these should be available in agent.params
def train_agent(agent, args, evaluate_only=False, multiple_predictions=False):
    model_dir = os.path.join(agent.params["performance_dir"], agent.name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    writer = SummaryWriter(os.path.join(model_dir, 'logs'))
    batch_size = args['batch_size']
    epochs = args['epochs']
    learning_rate = args['learning_rate']
    num_workers = args['num_worker']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find the mean and std of training data.
    dataset = AgentDataset(
        os.path.join(agent.params["performance_dir"], f"{agent.name}-performance-validation-set.csv"),
        img_dir=os.path.join(agent.params['dataset_dir'], "validation-set"),
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        sample_count=1e3
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    mean, std = get_mean_std(data_loader)
    print(f"Training set: mean: {mean.numpy()}, std: {std.numpy()}")

    transform = transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
    )

    dataset = AgentDataset(
        os.path.join(agent.params["performance_dir"], f"{agent.name}-performance-validation-set.csv"),
        img_dir=os.path.join(agent.params['dataset_dir'], "validation-set"),
        transform=transform,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    validation_set = AgentDataset(
        os.path.join(agent.params["performance_dir"], f"{agent.name}-performance-test-set.csv"),
        img_dir=os.path.join(agent.params['dataset_dir'], "test-set"),
        transform=transform,
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = None
    if args["model_name"] == "baseline":
        model = BaselineModel()
    elif args["model_name"] in ["resnet34", "resnet18"]:
        model = models.resnet34(
            pretrained=args["pretrained"]
        ) if args["model_name"] == "resnet34" else models.resnet18(
            pretrained=args["pretrained"]
        )

        if args["pretrained"]:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.5)),
            ('fc1', nn.Linear(512, 256)),
            ('activation1', nn.ReLU()),
            ('dropout2', nn.Dropout(0.3)),
            ('fc2', nn.Linear(256, 128)),
            ('activation2', nn.ReLU()),
            ('fc3', nn.Linear(128, 1))
        ]))

    elif args["model_name"] in ["resnet50", "resnet101", "resnet152", "wide_resnet101_2"]:
        if args["model_name"] == "resnet50":
            model = models.resnet50(pretrained=args['pretrained'])
        elif args["model_name"] == "resnet101":
            model = models.resnet101(pretrained=args['pretrained'])
        elif args["model_name"] == "resnet152":
            model = models.resnet152(pretrained=args['pretrained'])
        elif args["model_name"] == "wide_resnet101_2":
            model = models.wide_resnet101_2(pretrained=args['pretrained'])

        if args["pretrained"]:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Sequential(
            OrderedDict(
                [
                    ('dropout1', nn.Dropout(0.5)),
                    ('fc1', nn.Linear(2048, 1024)),
                    ('activation1', nn.ReLU()),
                    ('dropout2', nn.Dropout(0.3)),
                    ('fc2', nn.Linear(1024, 256)),
                    ('activation2', nn.ReLU()),
                    ('dropout3', nn.Dropout(0.3)),
                    ('fc3', nn.Linear(256, 128)),
                    ('activation3', nn.ReLU()),
                    ('fc4', nn.Linear(128, 1))
                ]
            )
        )
    elif args["model_name"] == "vgg19_bn":
        model = models.vgg19_bn(pretrained=args["pretrained"])

        if args["pretrained"]:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('activation1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(4096, 128)),
            ('activation2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(128, 1))
            # ('out', nn.Sigmoid())
        ]))

    else:
        raise NotImplementedError("Model not found")

    assert (model is not None)
    print(model)

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if evaluate_only:
        print("Evaluating on the test set...")
        checkpoint = torch.load(
            os.path.join(model_dir, f"checkpoint-best.pth.tar")
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        if multiple_predictions:
            prediction_count = 3
            print(f"Multiple predictions ({prediction_count}) with dropout activated.")
            # apply dropout during inference
            model.apply(apply_dropout)
            # TODO: Don't hardcode number of predictions
            # TODO: Remove duplicate code segment below
            dfs = []
            # TODO: Move this loop to where the model is spitting out the predictions.
            #  Make sure the output is not identical
            for i in range(1, prediction_count + 1):
                indices = []
                predictions = []
                with torch.no_grad():
                    for idx, image, label in tqdm(validation_loader, desc=f"Evaluating {agent.name}"):
                        inputs = image.to(device)
                        preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                        predictions += preds.tolist()
                        indices += idx
                dfs.append(
                    pd.DataFrame(
                        {
                            "image_id": indices,
                            f"proba_{i}": predictions
                        },
                    ).set_index('image_id')
                )

            all_dfs = pd.concat(dfs, axis=1)
            all_dfs['proba_mean'] = all_dfs.mean(axis=1)
            all_dfs['proba_var'] = all_dfs.var(axis=1)
            all_dfs.to_csv(
                os.path.join(model_dir, f'{agent.name}-prediction-probability-multi.csv'),
            )
        else:
            indices = []
            predictions = []
            with torch.no_grad():
                for idx, image, label in tqdm(validation_loader, desc=f"Evaluating {agent.name}"):
                    inputs = image.to(device)
                    preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                    predictions += preds.tolist()
                    indices += idx

            pd.DataFrame(
                {
                    'image_id': indices,
                    'proba': predictions
                }
            ).to_csv(
                os.path.join(model_dir, f'{agent.name}-prediction-probability.csv'),
                index=False
            )

        return

    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss()
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
            filename=os.path.join(model_dir, f'checkpoint-{epoch:03d}-val-{train_acc:.3f}.pth.tar')
        )

        # add to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        # writer.add_scalar("Loss/validation", val_loss, epoch)

        writer.flush()

    print('Finished Training.')


def evaluate(agent_group, assignment, assignment_type="optimized"):
    ind_eval = {}
    evaluation = {}
    total_acc, total_fbeta = 0., 0.
    for agent in agent_group.agents:
        # Read the test performance file
        performance_df = pd.read_csv(
            os.path.join(agent.params['performance_dir'], f"{agent.name}-performance-test-set.csv"),
            dtype={"image_id": str}
        )
        # individual performance
        acc = performance_df.performance.sum() / len(performance_df)
        fbeta = metrics.fbeta_score(
            performance_df.label.to_list(),
            performance_df.prediction.to_list(),
            beta=2.0
        )
        ind_eval[agent.name] = {
            "accuracy": f"{acc:.3f}",
            "fbeta": f"{fbeta:.3f}"
        }
        # assigned performance
        _df = performance_df[performance_df.image_id.isin(assignment[agent.name])]
        acc = _df.performance.sum() / len(_df)
        total_acc += acc
        fbeta = metrics.fbeta_score(_df.label.to_list(), _df.prediction.to_list(), beta=2.0)
        total_fbeta += fbeta
        evaluation[agent.name] = {
            "accuracy": f"{acc:.3f}",
            "fbeta": f"{fbeta:.3f}"
        }

    evaluation['avg_acc'] = f"{total_acc / len(agent_group.agents):.3f}"
    evaluation['avg_fbeta'] = f"{total_fbeta / len(agent_group.agents):.3f}"
    final_eval = {
        'individual': ind_eval,
        'assigned': evaluation
    }
    print(final_eval)
    with open(os.path.join(
            agent_group.agents[0].params['performance_dir'],
            # TODO: Remove hardcoded index, get params in a generalized way
            f"./{assignment_type}-assignment-performance.json"
    ), 'w') as f:
        json.dump(final_eval, f)


def main(args, train=True):
    agent_group = AgentGroup(AgentConfig.config_dir)
    # Train and evaluate to get the probabilities
    multiple_prediction = True
    for agent in agent_group.agents:
        print(f"Training and evaluating {agent}")
        if train:
            train_agent(agent, args)
        train_agent(agent, args, evaluate_only=True, multiple_predictions=multiple_prediction)

    # Probability matrix need to be in the following format:
    #         image_1 image_2 image_3 ... image_n
    # agent_1
    # agent_2
    # agent_3
    # ...
    # ...
    # agent_n
    #
    # TODO: Get the probabilities for all the datasets (we have agent-specific datasets)
    #  while doing so, normalize the test sets using dataset specific mean and std
    probabilities = pd.concat(
        [
            pd.read_csv(
                os.path.join(
                    agent.params['performance_dir'],
                    agent.name,
                    f"{agent.name}-prediction-probability{'-multi' if multiple_prediction else ''}.csv"
                ),
                usecols=['image_id', 'proba_mean'] if multiple_prediction else ['image_id', 'proba'],
                dtype={'image_id': str}
            ).set_index('image_id') for agent in agent_group.agents
        ],
        axis=1
    ).transpose().dropna(axis=1)  # Transpose is to get the matrix in the required format
    # TODO: Remove the above dropna operation and investigate what's wrong

    probabilities.to_csv(
        os.path.join(
            agent_group.agents[0].params['performance_dir'],
            "assignment_probabilities.csv"
        )
    )
    # Find optimized assignment
    assignment = []
    random_assignment = []
    for i in range(0, probabilities.shape[1], len(agent_group.agents)):
        df = probabilities.iloc[:, i:i + len(agent_group.agents)]
        img_ids = df.columns
        _agent_indices = linear_sum_assignment(df.to_numpy(), maximize=True)[1]
        assignment.append(img_ids[_agent_indices].tolist())
        img_ids = img_ids.tolist()
        np.random.shuffle(img_ids)
        random_assignment.append(img_ids)

    # TODO: Take care of the last element that has less than agent_count elements
    assignment = pd.DataFrame.from_records(
        np.stack(assignment[:-1], axis=0),
        columns=[agent.name for agent in agent_group.agents]
    )
    assignment.to_csv(
        os.path.join(agent_group.agents[0].params['performance_dir'], "assignment.csv"),
        index=False
    )
    random_assignment = pd.DataFrame.from_records(
        np.stack(random_assignment[:-1], axis=0),
        columns=[agent.name for agent in agent_group.agents]
    )

    print("Evaluating optimized assignment")
    evaluate(agent_group, assignment, assignment_type="optimized")
    print("Evaluating random assignment")
    evaluate(agent_group, random_assignment, assignment_type="random")


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a recommender model')
    parser.add_argument('--input', type=str, required=True, action='store',
                        help="JSON input")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate only")
    args_cmd = parser.parse_args()
    if not os.path.isfile(args_cmd.input):
        raise FileNotFoundError(f"Input {args_cmd.input} not found.")

    with open(args_cmd.input) as f:
        args = json.load(f)

    if args_cmd.evaluate:
        main(args, train=False)
    else:
        main(args, train=True)
