import os
from argparse import ArgumentParser
from functools import partial

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from agent import Agent
from agent.dataset import XrayImageDataset
from utils.utils import save_checkpoint, load_checkpoint, binary_acc, get_mean_std

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def train(agent, evaluate_only=True):
    writer = SummaryWriter(agent.model_dir)

    def send_stats(i, module, input, output):
        writer.add_scalar(f"{i}-mean", output.data.std())
        writer.add_scalar(f"{i}-stddev", output.data.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find the mean and std of training data using a subset of training data
    sample_size = 10_000
    print(f"Calculating mean and std of the training set on {sample_size} samples.")
    train_data = XrayImageDataset(
        annotations_file=os.path.join(agent.params["dataset_dir"], "train-labels.csv"),
        img_dir=os.path.join(agent.params["dataset_dir"], "train-set"),
        transform=transforms.Compose([transforms.ToTensor()]),
        sample_count=sample_size,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=agent.params["batch_size"],
        shuffle=True,
        num_workers=agent.params["num_worker"],
        pin_memory=True,
    )

    # use the mean and std to normalize
    mean, std = get_mean_std(train_loader)
    torch.save(
        {"mean": mean, "std": std}, os.path.join(agent.model_dir, "norm_data.pth")
    )
    print(f"Training data : mean: {mean} and std: {std}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    datasets = {
        "train": XrayImageDataset(
            annotations_file=os.path.join(
                agent.params["dataset_dir"], "train-labels.csv"
            ),
            img_dir=os.path.join(agent.params["dataset_dir"], "train-set"),
            transform=transform,
        ),
        "valid": XrayImageDataset(
            annotations_file=os.path.join(
                agent.params["dataset_dir"], "validation-labels.csv"
            ),
            img_dir=os.path.join(agent.params["dataset_dir"], "validation-set"),
            transform=transform,
        ),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=agent.params["batch_size"],
            shuffle=True,
            num_workers=agent.params["num_worker"],
            pin_memory=True,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=agent.params["batch_size"],
            shuffle=True,
            num_workers=agent.params["num_worker"],
            pin_memory=True,
        ),
    }

    model = agent.model

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # writer.add_graph(model, torch.rand([1, 3, 224, 224]))
    start_epoch = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=agent.params["learning_rate"])
    model, optimizer, start_epoch = load_checkpoint(
        model, optimizer, os.path.join(agent.model_dir, "checkpoint-best.pth.tar")
    )

    best_accuracy = 0.0
    for epoch in range(start_epoch, agent.params["epochs"] + 1):
        epoch_loss, epoch_acc = 0.0, 0.0
        running_loss, running_acc = 0.0, 0.0
        model.train()
        for i, (image_ids, images, labels) in enumerate(loaders["train"], 0):
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
            for i, (image_ids, images, labels) in enumerate(loaders["valid"], 0):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                # add to tensorboard
                # grid = torchvision.utils.make_grid(images[:8])
                # writer.add_image('validation/images', grid, epoch)

                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = binary_acc(outputs, labels)
                val_accuracy += acc.item()
                val_loss += loss.item()

        val_accuracy = val_accuracy / len(loaders["valid"])
        val_loss = val_loss / len(loaders["valid"])
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
                agent.model_dir,
                f"checkpoint-{epoch:03d}-val-{val_accuracy:.3f}.pth.tar",
            ),
        )

        train_loss = epoch_loss / len(loaders["train"])
        train_acc = epoch_acc / len(loaders["train"])
        print(
            f"Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc:"
            f" {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {best_accuracy:.3f}"
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a model in xray images")
    parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Evaluate only"
    )
    args = parser.parse_args()

    #TODO: Mofify the following ugly segment
    agent_one = Agent("agent_one", "./configs/baseline.json")
    train(agent_one)
    agent_two = Agent("agent_two", "./configs/resnet18.json")
    train(agent_two)
    agent_three = Agent("agent_three", "./configs/resnet34.json")
    train(agent_three)
    agent_four = Agent("agent_four", "./configs/resnet50.json")
    train(agent_four)
    agent_five = Agent("agent_five", "./configs/resnet101.json")
    train(agent_five)
    agent_six = Agent("agent_six", "./configs/resnet152.json")
    train(agent_six)
