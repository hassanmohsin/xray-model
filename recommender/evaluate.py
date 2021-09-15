import os
from collections import defaultdict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from recommender.dataset import AgentDataset
from recommender.performance import apply_dropout
from xray.agent import AgentGroup
from xray.config import AgentConfig
from xray.dataset import XrayImageDataset


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def load_agent_dataloaders(agent_group, set, sample_count=None, shuffle=False, pin_memory=True):
    loaders = {}
    for agent in agent_group.agents:
        # Load normalization data
        norm_data_file = os.path.join(agent.params["agent_dir"], "norm_data.pth")
        assert os.path.isfile(norm_data_file)
        norm_data = torch.load(norm_data_file)

        # Create dataset
        loaders[agent.name] = DataLoader(
            XrayImageDataset(
                annotations_file=os.path.join(agent.params["dataset_dir"], f"{set}-labels.csv"),
                img_dir=os.path.join(agent.params["dataset_dir"], f"{set}-set"),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=norm_data['mean'],
                        std=norm_data['std']
                    )
                ]),
                sample_count=sample_count
            ),
            batch_size=agent.params['batch_size'],
            shuffle=shuffle,
            num_workers=agent.params['num_worker'],
            pin_memory=pin_memory
        )

    return loaders


def load_performance_dataloaders(agent_group, set, sample_count=None, shuffle=False, pin_memory=True):
    loaders = {}
    for agent in agent_group:
        # Load normalization data (mean, std)
        norm_data_file = os.path.join(agent.params["agent_dir"], "norm_data.pth")
        assert os.path.isfile(norm_data_file)
        norm_data = torch.load(norm_data_file)

        # Create the dataset
        loaders[agent.name] = DataLoader(
            AgentDataset(
                os.path.join(agent.params["performance_dir"], f"{agent.name}-performance-{set}-set.csv"),
                img_dir=os.path.join(agent.params['dataset_dir'], f"{set}-set"),
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=norm_data[agent.name]['mean'],
                            std=norm_data[agent.name]['std']
                        )
                    ]
                ),
                sample_count=sample_count
            ),
            batch_size=agent.params['batch_size'],
            shuffle=shuffle,
            num_workers=agent.params['num_worker'],
            pin_memory=pin_memory
        )

    return dict(zip([agent.name for agent in agent_group.agents], loaders))


def evaluate(agent, dataloader, dataset_name, prediction_count):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = agent.model
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    checkpoint = torch.load(
        os.path.join(agent.params["agent_dir"], f"checkpoint-best.pth.tar")
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if prediction_count > 1:
        print(f"Multiple predictions ({prediction_count}) with dropout activated.")
        # apply dropout during inference
        model.apply(apply_dropout)

    indices, labels = predictions = [], [], defaultdict(list)
    with torch.no_grad():
        for idx, image, label in tqdm(dataloader, desc=f"Evaluating {agent.name} "):
            inputs = image.to(device)
            labels += label.numpy().tolist()
            preds = [torch.sigmoid(model(inputs)).squeeze().cpu().numpy() for _ in range(prediction_count)]
            for i, pred in enumerate(preds):
                predictions[f"proba_{i}"] += pred.tolist()
            indices += idx

    pred_df = pd.DataFrame()
    pred_df["image_id"] = [dataset_name + '-' + i for i in indices]
    pred_df["label"] = pd.Series(labels)
    for k, v in predictions.items():
        pred_df[k] = pd.Series(v)
    pred_df["proba_mean"] = pred_df.mean(axis=1, numeric_only=True)
    pred_df["proba_var"] = pred_df.var(axis=1, numeric_only=True)

    return pred_df


def main():
    agent_group = AgentGroup(AgentConfig.config_dir)

    # Get the predictions from the agents for all the datasets
    for set_name in ["validation", "test"]:
        dataloaders = load_agent_dataloaders(agent_group, set=set_name, sample_count=None)
        for agent in agent_group.agents:
            for dataset_name, loader in dataloaders.items():
                df = evaluate(agent, loader, dataset_name, prediction_count=3)
                df.to_csv(
                    os.path.join(agent.params["agent_dir"], f"{agent.name}_on_{dataset_name}-{set_name}.csv"),
                    index=False
                )
            # TODO: merge the dfs for each agent and write in one single file

    # Get the prediction from the predictors for all the datasets


if __name__ == '__main__':
    main()
