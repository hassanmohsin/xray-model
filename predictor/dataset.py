import os

import pandas as pd
import torch
from PIL import Image as Im
from torch.utils.data import Dataset

from agent import Agent, AgentGroup


class PredictorDataset(Dataset):
    def __init__(self, agent, transform, subset="validation", sample_count=None):
        self.agent = agent
        self.agent_name = agent.name
        self.transform = transform
        agent_group = AgentGroup()
        self.img_dirs = dict(zip(
            [agent.name for agent in agent_group.agents],
            [os.path.join(agent.params['dataset_dir'], "validation-set") for agent in agent_group.agents]
        ))
        self.data = pd.read_csv(
            os.path.join(self.agent.model_dir, f"{self.agent.name}-predictions-on-{subset}-set.csv"),
            nrows=sample_count
        )
        self.data['pred_label'] = self.data.proba_mean.apply(lambda x: round(x))
        self.data['success'] = self.data.apply(lambda row: int(row['label'] == row['pred_label']), axis=1)
        self.data['dataset_name'] = self.data.image_id.apply(lambda x: x.split('-')[0])
        self.data['image_id'] = self.data.image_id.copy().apply(lambda x: x.split('-')[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Im.open(
            os.path.join(
                self.img_dirs[row['dataset_name']],
                f"{row['image_id']}.png"
            )
        ).convert('RGB')
        img = self.transform(img)

        return '-'.join([row['dataset_name'], row['image_id']]), img, torch.tensor(row['success'], dtype=torch.float)
# class AgentDataset(Dataset):
#     def __init__(self, performance_file, img_dir, transform, sample_count=None):
#         self.performance = pd.read_csv(
#             performance_file,
#             dtype={"image_id": str},
#             nrows=sample_count
#         )
#         self.img_dir = img_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.performance)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.performance.image_id[idx].split('-')[1] + ".png")
#         image = Im.open(img_path).convert('RGB')
#         image = self.transform(image)
#         label = round(self.performance['proba_mean'][idx])
#
#         return self.performance.image_id[idx], image, torch.tensor(label)

#
# class AgentDataset(Dataset):
#     def __init__(self, performance_file, img_dir, transform, sample_count=None):
#         self.performance = pd.read_csv(
#             performance_file,
#             dtype={"image_id": str},
#             nrows=sample_count
#         )
#         self.img_dir = img_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.performance)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.performance.image_id[idx] + ".png")
#         image = Im.open(img_path).convert('RGB')
#         image = self.transform(image)
#         label = self.performance['performance'][idx]
#
#         return self.performance.image_id[idx], image, torch.tensor(label)
