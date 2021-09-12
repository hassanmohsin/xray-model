import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from recommender.dataset import XrayImageDataset
from xray.agent import AgentGroup

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def apply_dropout(m):
    for each_module in m.children():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
    return m


def get_preds(agent, transform, device, dataset_type):
    output_file = os.path.join(agent.params["performance_dir"], f"{agent.name}-performance-{dataset_type}-set.csv")
    if os.path.isfile(output_file):
        print(f"ERROR: {output_file} file exists!")
        return

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(agent.model)
    else:
        model = agent.model

    model.to(device)
    checkpoint_file = os.path.join(agent.params['agent_dir'], "checkpoint-best.pth.tar")
    if not os.path.isfile(checkpoint_file):
        print(f"ERROR: {checkpoint_file} not found.")
        return

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.apply(apply_dropout)

    if not os.path.isdir(agent.params["dataset_dir"]):
        print(f"ERROR: {agent.params['dataset_dir']} not found.")
        return

    dataset = XrayImageDataset(
        annotations_file=os.path.join(agent.params['dataset_dir'], f"{dataset_type}-labels.csv"),
        img_dir=os.path.join(agent.params['dataset_dir'], f"{dataset_type}-set"),
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=agent.params['batch_size'],
        shuffle=False,
        num_workers=agent.params['num_worker'],
        pin_memory=True
    )

    ground_truth = []
    predictions = []
    probabilities = []
    success_labels = []
    image_ids = []
    with torch.no_grad():
        for image_id, image, labels in tqdm(dataloader, desc=f"{agent.name} on {dataset_type} set: "):
            pred = model(image.to(device))
            proba = torch.sigmoid(pred).squeeze().cpu()
            pred = proba.round()
            ground_truth += labels.to(torch.int).tolist()
            predictions += pred.to(torch.int).tolist()
            probabilities += proba.to(torch.float).numpy().round(4).tolist()
            success_labels += (pred == labels).to(torch.int).tolist()
            image_ids += image_id

    pd.DataFrame.from_dict(
        {
            'image_id': image_ids,
            'label': ground_truth,
            'probability': probabilities,
            'prediction': predictions,
            'performance': success_labels
        }
    ).to_csv(
        output_file,
        index=False
    )


if __name__ == '__main__':
    # TODO: remove hardcoded normalization data. Read from `normalization-data.json`
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.7165, 0.7446, 0.7119],
            [0.3062, 0.2433, 0.2729]
        )
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    agent_group = AgentGroup("./configs")

    for agent in agent_group.agents:
        for set in ['validation', 'test']:
            get_preds(agent, transform, device, set)
