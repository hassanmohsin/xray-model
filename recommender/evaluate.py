import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from recommender.dataset import XrayImageDataset
from xray.models import BaselineModel, ModelOne, ModelTwo

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def get_preds(model, checkpoint, transform, device):
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = XrayImageDataset(f"{dataset_dir}/test-labels.csv", f"{dataset_dir}/test-set", transform)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=56,
        pin_memory=True
    )

    pred_labels = []
    image_ids = []
    with torch.no_grad():
        for i, (image_id, image, labels) in enumerate(dataloader, 0):
            pred = model(image.to(device))
            # pred_labels += torch.round(torch.sigmoid(pred)).squeeze().to(torch.int).cpu().numpy().tolist()
            pred_labels += torch.sigmoid(pred).squeeze().cpu().numpy().tolist()
            image_ids += image_id

    return image_ids, pred_labels


def agent_one(device):
    checkpoint = "../saved_models/baseline/checkpoint.pth.tar"
    model = BaselineModel().to(device)
    # if torch.cuda.is_available() > 1:
    #     model = nn.DataParallel(model)
    #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return get_preds(model, checkpoint, transform, device)


def agent_two(device):
    checkpoint = "../saved_models/ModelOne/checkpoint.pth.tar"
    model = ModelOne().to(device)
    # if torch.cuda.is_available() > 1:
    #     model = nn.DataParallel(model)
    #
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return get_preds(model, checkpoint, transform, device)


def agent_three(device):
    checkpoint = "../saved_models/ModelTwo/checkpoint.pth.tar"
    model = ModelTwo().to(device)
    # if torch.cuda.is_available() > 1:
    #     model = nn.DataParallel(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return get_preds(model, checkpoint, transform, device)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dataset_dir = "../dummy-dataset"
    agent_one_img_ids, agent_one_probs = agent_one(device)
    agent_two_img_ids, agent_two_probs = agent_two(device)
    agent_three_img_ids, agent_three_probs = agent_three(device)

    # Check the order of the images
    for (i, j, k) in zip(agent_one_img_ids, agent_two_img_ids, agent_three_img_ids):
        assert i == j == k, "Image indices are not in order."

    pd.DataFrame.from_dict(
        {
            'image_id': agent_one_img_ids,  # TODO: Make sure the indices are in order for all agents
            'agent_one': agent_one_probs,
            'agent_two': agent_two_probs,
            'agent_three': agent_three_probs
        }
    ).to_csv(
        "probabilities.csv",
        index=False
    )
