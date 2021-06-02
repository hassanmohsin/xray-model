import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class XrayImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, test_data=False, transform=None):
        self.img_labels = pd.read_csv(annotations_file, dtype={'ImageId': str, 'Label': int})
        self.img_dir = img_dir
        self.test_data = test_data
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.test_data:
            return self.img_labels.iloc[idx, 0], image
        else:
            return image, torch.tensor(label)
