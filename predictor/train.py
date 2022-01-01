import json
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

seed = 42
# torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def train(args, agent, evaluate=False, resume_training=False, multiple_predictions=3):
    # args is for predictor config, agent is what the predictor is for.
    model_dir = os.path.join(AgentConfig.predictor_dir, agent.name, args["model_name"])
    best_checkpoint_file = os.path.join(model_dir, f"checkpoint-best.pth.tar")
    if not evaluate and os.path.isfile(best_checkpoint_file):
        print(f"{best_checkpoint_file} file exists. Skipping predictor training.")
        return

    writer = SummaryWriter(model_dir)

    def send_stats(i, module, input, output):
        writer.add_scalar(f"{i}-mean", output.data.std())
        writer.add_scalar(f"{i}-stddev", output.data.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find the mean and std of training data using a subset of training data
    norm_file = os.path.join(model_dir, "norm_data_predictor.pth")
    if os.path.isfile(norm_file):
        norm_data = torch.load(norm_file)
        mean, std = norm_data["mean"], norm_data["std"]
    else:
        sample_size = 100_000
        print(f"Calculating mean and std of the training set on {sample_size} samples.")
        validation_data = PredictorDataset(
            agent,
            subset="validation",
            transform=transforms.ToTensor(),
            sample_count=sample_size,
        )

        validation_loader = DataLoader(
            validation_data,
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args["num_worker"],
            pin_memory=True,
        )

        # use the mean and std to normalize
        mean, std = get_mean_std(validation_loader)
        torch.save({"mean": mean, "std": std}, norm_file)
    print(f"Training data : mean: {mean} and std: {std}")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    datasets = {
        "validation": PredictorDataset(agent, transform=transform, subset="validation"),
        "test": PredictorDataset(agent, transform=transform, subset="test"),
    }

    loaders = {
        "validation": DataLoader(
            datasets["validation"],
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args["num_worker"],
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args["num_worker"],
            pin_memory=True,
        ),
    }

    model = get_model(args["model_name"])

    if evaluate:
        print(f"Evaluating {agent.name} on the test set...")
        checkpoint = torch.load(os.path.join(model_dir, f"checkpoint-best.pth.tar"))
        model.load_state_dict(checkpoint["state_dict"])
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)
        model.eval()

        if multiple_predictions:
            print(
                f"Multiple predictions ({multiple_predictions}) with dropout activated."
            )
            # apply dropout during inference
            model.apply(apply_dropout)

            def predict_agent_dataset(loader):
                print(f"{args['model_name']} predicting the test set...")
                # TODO: Don't hardcode number of predictions
                # TODO: Remove duplicate code segment below
                # TODO: Move this loop to where the model is spitting out the predictions.
                #  Make sure the output is not identical
                indices, labels, predictions = [], [], defaultdict(list)
                with torch.no_grad():
                    for idx, image, label in tqdm(
                        loader, desc=f"Evaluating {args['model_name']} on test set"
                    ):
                        inputs = image.to(device)
                        labels += label.numpy().tolist()
                        preds = [
                            torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                            for _ in range(multiple_predictions)
                        ]
                        for i, pred in enumerate(preds):
                            predictions[f"proba_{i}"] += pred.tolist()
                        indices += idx

                pred_df = pd.DataFrame()
                pred_df["image_id"] = pd.Series(indices)
                pred_df["label"] = pd.Series(labels)
                for k, v in predictions.items():
                    pred_df[k] = pd.Series(v)
                pred_df["proba_mean"] = pred_df.mean(axis=1, numeric_only=True)
                pred_df["proba_var"] = pred_df.var(axis=1, numeric_only=True)

                return pred_df

            prediction_file = os.path.join(
                model_dir, f"{agent.name}-prediction-probability-test-set.csv"
            )
            if not os.path.isfile(prediction_file):
                predict_agent_dataset(loaders["test"]).to_csv(
                    prediction_file,
                    index=False,
                )
            else:
                print(f"{prediction_file} file exists. Skipping re-evaluation.")

        else:
            indices = []
            predictions = []
            with torch.no_grad():
                for idx, image, label in tqdm(
                    loaders["test"], desc=f"Evaluating {args['model_name']} on test set"
                ):
                    inputs = image.to(device)
                    preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                    predictions += preds.tolist()
                    indices += idx

            pd.DataFrame({"image_id": indices, "proba": predictions}).to_csv(
                os.path.join(
                    model_dir, f"{agent.name}-prediction-probability-test-set.csv"
                ),
                index=False,
            )

        return

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # writer.add_graph(model, torch.rand([1, 3, 224, 224]))
    start_epoch = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=agent.params["learning_rate"])
    if resume_training:
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, os.path.join(model_dir, "checkpoint-best.pth.tar")
        )

    best_accuracy = 0.0
    for epoch in range(
        start_epoch, args["epochs"] + 1
    ):  # loop over the dataset multiple times
        epoch_loss, epoch_acc = 0.0, 0.0
        running_loss, running_acc = 0.0, 0.0
        model.train()
        for i, (image_ids, images, labels) in enumerate(loaders["validation"], 0):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
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

        model.eval()
        val_accuracy = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for i, (image_ids, images, labels) in enumerate(loaders["test"], 0):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                # add to tensorboard
                # grid = torchvision.utils.make_grid(images[:8])
                # writer.add_image('validation/images', grid, epoch)

                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = binary_acc(outputs, labels)
                val_accuracy += acc.item()
                val_loss += loss.item()

        val_accuracy = val_accuracy / len(loaders["test"])
        val_loss = val_loss / len(loaders["test"])
        is_best = bool(val_accuracy > best_accuracy)
        best_accuracy = max(val_accuracy, best_accuracy)
        # Save checkpoint if is a new best
        save_checkpoint(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "state_dict": model.module.state_dict(),
                "loss": val_loss,
                "best_accuracy": best_accuracy,
            },
            is_best,
            filename=os.path.join(
                model_dir, f"checkpoint-{epoch:03d}-val-{val_accuracy:.3f}.pth.tar"
            ),
        )

        train_loss = epoch_loss / len(loaders["validation"])
        train_acc = epoch_acc / len(loaders["validation"])
        print(
            f"Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc:"
            f" {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_accuracy:.3f} | Best Acc: {best_accuracy:.3f}"
        )

        # add to tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/accuracy", val_accuracy, epoch)

        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f"{name}.grad", weight, epoch)

        for i, m in enumerate(model.children()):
            m.register_forward_hook(partial(send_stats, i))

        writer.flush()

    print("Finished Training.")


def evaluate(args, agent_group, assignment, assignment_type="optimized"):
    ind_eval = {}
    evaluation = {}
    total_acc, total_fbeta = 0.0, 0.0
    for agent in agent_group.agents:
        # Read the test performance file
        df = pd.read_csv(
            os.path.join(
                AgentConfig.predictor_dir,
                agent.name,
                args["model_name"],
                f"{agent.name}-prediction-probability-test-set.csv",
            ),
            dtype={"image_id": str},
        )
        df["pred_label"] = df.proba_mean.apply(lambda x: round(x))
        df["success"] = df.apply(
            lambda row: int(row["label"] == row["pred_label"]), axis=1
        )

        # individual performance
        acc = df.success.mean()
        fbeta = metrics.fbeta_score(
            df.label.to_list(), df.pred_label.to_list(), beta=2.0
        )
        ind_eval[agent.name] = {"accuracy": f"{acc:.3f}", "fbeta": f"{fbeta:.3f}"}
        # assigned performance
        _df = df[df.image_id.isin(assignment[agent.name])]
        acc = _df.success.mean()
        total_acc += acc
        fbeta = metrics.fbeta_score(
            _df.label.to_list(), _df.pred_label.to_list(), beta=2.0
        )
        total_fbeta += fbeta
        evaluation[agent.name] = {"accuracy": f"{acc:.3f}", "fbeta": f"{fbeta:.3f}"}

    evaluation["avg_acc"] = f"{total_acc / len(agent_group.agents):.3f}"
    evaluation["avg_fbeta"] = f"{total_fbeta / len(agent_group.agents):.3f}"
    final_eval = {"individual": ind_eval, "assigned": evaluation}
    print(final_eval)
    with open(
        os.path.join(
            AgentConfig.predictor_dir,
            # TODO: Remove hardcoded index, get params in a generalized way
            f"./{assignment_type}-assignment-performance.json",
        ),
        "w",
    ) as f:
        json.dump(final_eval, f)


def main(args, training=True):
    agent_group = AgentGroup()
    # Train and evaluate to get the probabilities
    multiple_prediction = 3
    for agent in agent_group.agents:
        print(f"Training and evaluating {agent.name}")
        # First train and then evaluate the predictor model
        train(args, agent, evaluate=False)
        train(args, agent, evaluate=True, multiple_predictions=multiple_prediction)

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
    probabilities = (
        pd.concat(
            [
                pd.read_csv(
                    os.path.join(
                        AgentConfig.predictor_dir,
                        agent.name,
                        args["model_name"],
                        f"{agent.name}-prediction-probability-test-set.csv",
                    ),
                    usecols=["image_id", "proba_mean"]
                    if multiple_prediction
                    else ["image_id", "proba"],
                    dtype={"image_id": str},
                ).set_index("image_id")
                for agent in agent_group.agents
            ],
            axis=1,
        )
        .transpose()
        .dropna(axis=1)
    )  # Transpose is to get the matrix in the required format
    # TODO: Remove the above dropna operation and investigate what's wrong

    probabilities.to_csv(
        os.path.join(AgentConfig.predictor_dir, "assignment_probabilities.csv")
    )
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
        columns=[agent.name for agent in agent_group.agents],
    )
    assignment.to_csv(
        os.path.join(AgentConfig.predictor_dir, "assignment.csv"), index=False
    )
    random_assignment = pd.DataFrame.from_records(
        np.stack(random_assignment[:-1], axis=0),
        columns=[agent.name for agent in agent_group.agents],
    )

    print("Evaluating optimized assignment")
    evaluate(args, agent_group, assignment, assignment_type="optimized")
    print("Evaluating random assignment")
    evaluate(args, agent_group, random_assignment, assignment_type="random")


def load_agent_dataloaders(
    agent_group, set, sample_count=None, shuffle=False, pin_memory=True
):
    loaders = {}
    for agent in agent_group.agents:
        # Load normalization data
        norm_data_file = os.path.join(agent.model_dir, "norm_data.pth")
        assert os.path.isfile(norm_data_file)
        norm_data = torch.load(norm_data_file)
        # Create dataset
        loaders[agent.name] = DataLoader(
            XrayImageDataset(
                annotations_file=os.path.join(
                    agent.params["dataset_dir"], f"{set}-labels.csv"
                ),
                img_dir=os.path.join(agent.params["dataset_dir"], f"{set}-set"),
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=norm_data["mean"], std=norm_data["std"]
                        ),
                    ]
                ),
                sample_count=sample_count,
            ),
            batch_size=agent.params["batch_size"],
            shuffle=shuffle,
            num_workers=agent.params["num_worker"],
            pin_memory=pin_memory,
        )

    return loaders


def evaluate_agent(agent, dataloader, dataset_name, prediction_count):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = agent.model
    checkpoint = torch.load(
        os.path.join(agent.model_dir, f"checkpoint-best-multi.pth.tar")
    )
    model.load_state_dict(checkpoint["state_dict"])
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    if prediction_count > 1:
        print(f"Multiple predictions ({prediction_count}) with dropout activated.")
        # apply dropout during inference
        model.apply(apply_dropout)

    indices, labels, predictions = [], [], defaultdict(list)
    with torch.no_grad():
        for idx, image, label in tqdm(
            dataloader, desc=f"Evaluating {agent.name} on {dataset_name} "
        ):
            inputs = image.to(device)
            labels += label.numpy().tolist()
            preds = [
                torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                for _ in range(prediction_count)
            ]
            for i, pred in enumerate(preds):
                predictions[f"proba_{i}"] += pred.tolist()
            indices += idx

    pred_df = pd.DataFrame()
    pred_df["image_id"] = [dataset_name + "-" + i for i in indices]
    pred_df["label"] = pd.Series(labels)
    for k, v in predictions.items():
        pred_df[k] = pd.Series(v)
    pred_df["proba_mean"] = pred_df.mean(axis=1, numeric_only=True)
    pred_df["proba_var"] = pred_df.var(axis=1, numeric_only=True)

    return pred_df


def get_predictions_from_agents(multiple_predictions=3):
    agent_group = AgentGroup()

    # Get the predictions from the agents for all the datasets
    for set_name in ["validation", "test"]:
        dataloaders = load_agent_dataloaders(
            agent_group, set=set_name, sample_count=None
        )
        for agent in agent_group.agents:
            combined_output_file = os.path.join(
                agent.model_dir, f"{agent.name}-predictions-on-{set_name}-set.csv"
            )
            if os.path.isfile(combined_output_file):
                print(
                    f"{combined_output_file} file exists. Skipping the prediction from {agent.name}."
                )
                continue
            dfs = []
            for dataset_name, loader in dataloaders.items():
                output_file = os.path.join(
                    agent.model_dir, f"{agent.name}_on_{dataset_name}-{set_name}.csv"
                )
                if os.path.isfile(output_file):
                    dfs.append(pd.read_csv(output_file, dtype={"label": int}))
                else:
                    df = evaluate_agent(
                        agent,
                        loader,
                        dataset_name,
                        prediction_count=multiple_predictions,
                    )
                    df.to_csv(output_file, index=False)
                    dfs.append(df)
            pd.concat(dfs).to_csv(
                combined_output_file,
                index=False,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a recommender model")
    parser.add_argument(
        "--input", type=str, required=True, action="store", help="JSON input"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    args_cmd = parser.parse_args()
    if not os.path.isfile(args_cmd.input):
        raise FileNotFoundError(f"Input {args_cmd.input} not found.")

    with open(args_cmd.input) as f:
        args = json.load(f)

    # get agent predictions on the validation and test set
    get_predictions_from_agents()

    if args_cmd.evaluate:
        main(args, training=False)
    else:
        main(args, training=True)
