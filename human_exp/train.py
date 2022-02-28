import json
from human_exp.utils import Mapping
import os
from argparse import ArgumentParser
from functools import partial
from collections import defaultdict
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

from agent import AgentGroup
from agent.config import AgentConfig
from agent.models import get_model
from agent.dataset import XrayImageDataset
from predictor.dataset import PredictorDataset
from utils.utils import (
    get_mean_std,
    save_checkpoint,
    load_checkpoint,
    binary_acc,
    apply_dropout,
)
from human_exp.dataset import HumanDataset


def evaluate(args, mapping, assignment, assignment_type="optimized"):
    ind_eval = {}
    evaluation = {}
    total_acc, total_fbeta = 0.0, 0.0
    agents = mapping.get_agents()
    performance = mapping.answers_df
    for agent in agents:
        # Read the test performance file
        df = performance[performance.name == agent]
        df["image_id"] = df.image_id.copy().apply(lambda x: f"img-{x}")

        # individual performance
        acc = df.correct.mean()
        fbeta = metrics.fbeta_score(df.label.to_list(), df.pred.to_list(), beta=1.0)
        ind_eval[agent] = {"accuracy": f"{acc:.3f}", "fbeta": f"{fbeta:.3f}"}
        # assigned performance
        _df = df[df.image_id.isin(assignment[agent])]
        acc = _df.correct.mean()
        total_acc += acc
        fbeta = metrics.fbeta_score(_df.label.to_list(), _df.pred.to_list(), beta=1.0)
        total_fbeta += fbeta
        evaluation[agent] = {"accuracy": f"{acc:.3f}", "fbeta": f"{fbeta:.3f}"}

    evaluation["avg_acc"] = f"{total_acc / len(agents):.3f}"
    evaluation["avg_fbeta"] = f"{total_fbeta / len(agents):.3f}"
    final_eval = {"individual": ind_eval, "assigned": evaluation}
    print(final_eval)
    # TODO: Remove hardcoded index, get params in a generalized way
    with open(
        os.path.join(
            "./human-models",
            f"./{assignment_type}-assignment-performance.json",
        ),
        "w",
    ) as f:
        json.dump(final_eval, f)


def retrain(args, predictor, human_agent_name, mapping, evaluate=True):
    output_dir = os.path.join("./human-models", human_agent_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(
        AgentConfig.predictor_dir, predictor.name, args["model_name"]
    )
    model = get_model(args["model_name"])

    # Find the mean and std of training data using a subset of training data
    norm_file = os.path.join(model_dir, "norm_data_predictor.pth")
    if os.path.isfile(norm_file):
        norm_data = torch.load(norm_file)
        mean, std = norm_data["mean"], norm_data["std"]
    else:
        print("norm_data_predictor.pth not found. Quiting...")
        return
    print(f"Training data : mean: {mean} and std: {std}")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    datasets = {
        "train": HumanDataset(
            human_agent_name, transform, mapping, subset="train", split=0.7
        ),
        "test": HumanDataset(
            human_agent_name, transform, mapping, subset="test", split=0.7
        ),
    }
    batch_size = 4
    num_workers = 2
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if evaluate:
        print(f"Evaluating {human_agent_name} on the test set...")
        checkpoint = torch.load(os.path.join(output_dir, f"checkpoint-best.pth.tar"))
        model.load_state_dict(checkpoint["state_dict"])
        # if torch.cuda.device_count() > 1:
        #     print("Using ", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        model.to(device)
        model.eval()

        indices = []
        predictions = []
        with torch.no_grad():
            for idx, image, label in tqdm(
                loaders["test"], desc=f"Evaluating {human_agent_name} on test set"
            ):
                inputs = image.to(device)
                preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                predictions += preds.tolist()
                indices += [f"img-{str(i)}" for i in idx.tolist()]

        pd.DataFrame({"image_id": indices, "proba": predictions}).to_csv(
            os.path.join(
                output_dir,
                f"{human_agent_name}-prediction-probability-test-set.csv",
            ),
            index=False,
        )

        return

    # Train the model
    model_weights_file = os.path.join(output_dir, f"checkpoint-best.pth.tar")
    if os.path.isfile(model_weights_file):
        print("Model is already trained. Quiting...")
        return

    learning_rate = 0.00001
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Load the best checkpoint from the previous training on synthetic data
    checkpoint = torch.load(os.path.join(model_dir, f"checkpoint-best.pth.tar"))
    model.load_state_dict(checkpoint["state_dict"])

    # Freeze the conv weights of the predictor model
    model.conv1.weight.requires_grad = False
    model.conv2.weight.requires_grad = False
    model.conv3.weight.requires_grad = False

    # if torch.cuda.device_count() > 1:
    #     print("Using ", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)

    model.train()
    best_acc = 0.0
    for epoch in range(50):
        epoch_loss, epoch_acc = 0.0, 0.0
        running_loss, running_acc = 0.0, 0.0
        for i, (idx, images, labels) in enumerate(loaders["train"]):
            images, labels = images.to(device), labels.to(device)
            # add to tensorboard
            # grid = torchvision.utils.make_grid(images[:8])
            # writer.add_image('train/images', grid, epoch)
            # writer.add_graph(model, inputs)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = binary_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            running_loss += loss.item()
            running_acc += acc.item()
            if i % 10 == 0:
                print(
                    f"iter: {i:04}: Running loss: {running_loss / 10:.3f} | Running acc: {running_acc / 10:.3f}"
                )
                running_loss = 0.0
                running_acc = 0.0

        train_acc = epoch_acc / len(loaders["train"])
        train_loss = epoch_loss / len(loaders["train"])
        is_best = bool(train_acc > best_acc)
        best_acc = max(best_acc, train_acc)
        print(
            f"Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc:"
            f" {train_acc:.3f} | Best Acc: {best_acc:.3f}"
        )
        save_checkpoint(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                # "state_dict": model.module.state_dict(),
                "state_dict": model.state_dict(),
                "loss": epoch_loss,
                "best_accuracy": best_acc,
            },
            is_best,
            filename=os.path.join(
                output_dir, f"checkpoint-{epoch:03d}-val-{train_acc:.3f}.pth.tar"
            ),
        )


def main(args):
    agent_group = AgentGroup()
    mapping = Mapping(
        ["./human_exp/exp1.json", "./human_exp/exp2.json", "./human_exp/exp3.json"],
        len(agent_group.agents),
        type="best",
    )
    for (predictor, human_agent_name) in zip(agent_group.agents, mapping.get_agents()):
        print(f"Retraining {predictor.name} for {human_agent_name}")
        # First train and then evaluate the predictor model
        retrain(args, predictor, human_agent_name, mapping, evaluate=False)
        retrain(args, predictor, human_agent_name, mapping, evaluate=True)

    probabilities = (
        pd.concat(
            [
                pd.read_csv(
                    os.path.join(
                        "./human-models",
                        agent,
                        f"{agent}-prediction-probability-test-set.csv",
                    ),
                    usecols=["image_id", "proba"],
                    dtype={"image_id": str},
                ).set_index("image_id")
                for agent in mapping.get_agents()
            ],
            axis=1,
        )
        .transpose()
        .dropna(axis=1)
    )

    probabilities.to_csv(os.path.join("./human-models", "assignment_probabilities.csv"))
    # Find optimized assignment
    assignment = []
    random_assignment = []
    for i in range(0, probabilities.shape[1], len(agent_group.agents)):
        df = probabilities.iloc[:, i : i + len(agent_group.agents)]
        img_ids = df.columns
        _agent_indices = linear_sum_assignment(df.to_numpy(), maximize=True)[1]
        assignment.append(img_ids[_agent_indices].tolist())
        img_ids = img_ids.tolist()
        np.random.shuffle(img_ids)
        random_assignment.append(img_ids)

    # TODO: Take care of the last element that has less than agent_count elements
    assignment = pd.DataFrame.from_records(
        np.stack(assignment[:-1], axis=0),
        columns=[agent for agent in mapping.get_agents()],
    )
    assignment.to_csv(os.path.join("./human-models", "assignment.csv"), index=False)
    random_assignment = pd.DataFrame.from_records(
        np.stack(random_assignment[:-1], axis=0),
        columns=[agent for agent in mapping.get_agents()],
    )

    print("Evaluating optimized assignment")
    evaluate(args, mapping, assignment, assignment_type="optimized")
    print("Evaluating random assignment")
    evaluate(args, mapping, random_assignment, assignment_type="random")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Retrain a predictor model based on human experiment data"
    )
    parser.add_argument(
        "--input", type=str, required=True, action="store", help="JSON input"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    args_cmd = parser.parse_args()
    if not os.path.isfile(args_cmd.input):
        raise FileNotFoundError(f"Input {args_cmd.input} not found.")

    with open(args_cmd.input) as f:
        args = json.load(f)

    if args_cmd.evaluate:
        main(args)
    else:
        main(args)
