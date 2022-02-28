import os
import pandas as pd
import torch
from PIL import Image as Im
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from agent import Agent, AgentGroup
from human_exp.utils import read_json, Mapping
from sklearn.model_selection import train_test_split


class HumanDataset(Dataset):
    """
    Builds dataset to retrain the predictor models using the dataset used for the human agents.
    X: Images
    Y: Predictions made by the agents (not the actual image labels)

    """

    def __init__(self, human_name, transform, mapping, subset="train", split=0.7):
        """
        agent_name: Name of the agent
        transform: Transformation to apply to the images
        json_files: List of json files containing the responses of the agents
        subset: Subset of the dataset to use (train, val, test)
        split: Percentage of the dataset to use for training
        """
        self.human_name = human_name
        self.transform = transform
        self.subset = subset
        self.answers_dict = mapping.answers_dict
        self.answers_df = mapping.answers_df
        self.performance = mapping.performance
        self.data = self.answers_df[self.answers_df.name == self.human_name]
        dataset_labels = self.data.dataset.to_frame()
        y = self.data.pred.to_frame()
        # X = self.data.image_id.to_frame()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, y, stratify=dataset_labels, test_size=0.4, random_state=42
        )
        self.X = self.X_train if self.subset == "train" else self.X_test
        self.y = self.y_train if self.subset == "train" else self.y_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        img = Im.open(
            os.path.join("./human_exp/experiment-images/images", f"img-{row.image_id}.png")
        ).convert("RGB")
        img = self.transform(img)

        return (
            row.image_id,
            img,
            torch.tensor(self.y.iloc[idx], dtype=torch.float),
        )


if __name__ == "__main__":
    json_files = [
        "./human_exp/exp1.json",
        "./human_exp/exp2.json",
        "./human_exp/exp3.json",
    ]
    agent_group = AgentGroup()
    agent = agent_group.agents[0]
    dataset = PredictorReTrainDataset(
        agent_name, transform=None, subset="train", json_files=json_files
    )
    # print(dataset.performance)

    map = Mapping(dataset.performance, 6)
    agents = map.get_agents()
