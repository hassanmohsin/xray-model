import os

import pandas as pd
import torch
import torchvision
from PIL import Image as Im
from torch.utils.data import Dataset, DataLoader


class MultiViewDataset(Dataset):
    def __init__(self, labels, img_dir, transform, test_data=False, sample_count=None):
        self.img_labels = pd.read_csv(labels, dtype={'ImageId': str, 'Label': int}, nrows=sample_count)
        self.img_dir = img_dir
        self.test_data = test_data
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_paths = [
            os.path.join(os.path.join(self.img_dir, view), self.img_labels.iloc[idx, 0] + ".png") for view in
            ['xview', 'yview', 'zview']
        ]
        images = [Im.open(img_path).convert('RGB') for img_path in img_paths]
        if isinstance(self.transform, list):
            images = [transform(image) for (transform, image) in zip(self.transform, images)]
        else:
            images = [self.transform(image) for image in images]
        label = self.img_labels.iloc[idx, 1]
        if self.test_data:
            return idx, self.img_labels.iloc[idx, 0], images
        else:
            return idx, images, torch.tensor(label)


if __name__ == '__main__':
    dataset_dir = "/data2/mhassan/xray/new-dataset"
    dataset = MultiViewDataset(
        labels=os.path.join(dataset_dir, "test-labels.csv"),
        img_dir=os.path.join(dataset_dir, "test-set"),
        transform=torchvision.transforms.ToTensor()
    )

    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for _, images, label in data_loader:
        print([img.size() for img in images])
        print(label)

        break
