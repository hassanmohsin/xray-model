import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from recommender.dataset import XrayImageDataset
from xray.models import ModelTwo


class Agent:
    def __init__(self, dataset_dir, model_dir):
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir

    # TODO: specify dataset type e.g., 'validation' or 'test'
    def get_preds(  model, checkpoint, transform, device):
        # TODO: get predictions from all models concurrently
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        dataset = XrayImageDataset(
            annotations_file=f"{self.dataset_dir}/validation-labels.csv",
            img_dir=f"{self.dataset_dir}/validation-set",
            transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        pred_labels = []
        image_ids = []
        with torch.no_grad():
            for image_id, image, labels in tqdm(dataloader, desc="Getting labels: "):
                pred = model(image.to(device))
                pred = torch.round(torch.sigmoid(pred)).squeeze().cpu()
                pred_labels += (pred == labels).to(torch.int).tolist()
                image_ids += image_id

        return image_ids, pred_labels


    def agent_one(device, transform):
        checkpoint = os.path.join(model_dir, "ModelTwo/checkpoint-best.pth.tar")
        model = ModelTwo()
        return get_preds(model, checkpoint, transform, device)


    def agent_two(device, transform):
        checkpoint = os.path.join(model_dir, "resnet18/checkpoint-best.pth.tar")
        model = models.resnet18(pretrained=False)
        layers_resnet = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.5)),
            ('fc1', nn.Linear(512, 256)),
            ('activation1', nn.ReLU()),
            ('dropout2', nn.Dropout(0.3)),
            ('fc2', nn.Linear(256, 128)),
            ('activation2', nn.ReLU()),
            ('fc3', nn.Linear(128, 1))
            # ('out', nn.Sigmoid())
        ]))

        model.fc = layers_resnet
        return get_preds(model, checkpoint, transform, device)


    def agent_three(device, transform):
        checkpoint = os.path.join(model_dir, "resnet34/checkpoint-best.pth.tar")
        model = models.resnet34(pretrained=False)
        layers_resnet = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.5)),
            ('fc1', nn.Linear(512, 256)),
            ('activation1', nn.ReLU()),
            ('dropout2', nn.Dropout(0.3)),
            ('fc2', nn.Linear(256, 128)),
            ('activation2', nn.ReLU()),
            ('fc3', nn.Linear(128, 1))
            # ('out', nn.Sigmoid())
        ]))

        model.fc = layers_resnet
        return get_preds(model, checkpoint, transform, device)


    def agent_four(device, transform):
        checkpoint = os.path.join(model_dir, "resnet50/checkpoint-best.pth.tar")
        model = models.resnet50(pretrained=False)

        layers_resnet = nn.Sequential(
            OrderedDict(
                [
                    ('dropout1', nn.Dropout(0.5)),
                    ('fc1', nn.Linear(2048, 1024)),
                    ('activation1', nn.ReLU()),
                    ('dropout2', nn.Dropout(0.3)),
                    ('fc2', nn.Linear(1024, 256)),
                    ('activation2', nn.ReLU()),
                    ('dropout3', nn.Dropout(0.3)),
                    ('fc3', nn.Linear(256, 128)),
                    ('activation3', nn.ReLU()),
                    ('fc4', nn.Linear(128, 1))
                ]
            )
        )

        model.fc = layers_resnet
        return get_preds(model, checkpoint, transform, device)


    def agent_five(device, transform):
        checkpoint = os.path.join(model_dir, "resnet101/checkpoint-best.pth.tar")
        model = models.resnet101(pretrained=False)

        layers_resnet = nn.Sequential(
            OrderedDict(
                [
                    ('dropout1', nn.Dropout(0.5)),
                    ('fc1', nn.Linear(2048, 1024)),
                    ('activation1', nn.ReLU()),
                    ('dropout2', nn.Dropout(0.3)),
                    ('fc2', nn.Linear(1024, 256)),
                    ('activation2', nn.ReLU()),
                    ('dropout3', nn.Dropout(0.3)),
                    ('fc3', nn.Linear(256, 128)),
                    ('activation3', nn.ReLU()),
                    ('fc4', nn.Linear(128, 1))
                ]
            )
        )

        model.fc = layers_resnet
        return get_preds(model, checkpoint, transform, device)


    def agent_six(device, transform):
        checkpoint = os.path.join(model_dir, "wide_resnet101_2/checkpoint-best.pth.tar")
        model = models.wide_resnet101_2(pretrained=False)

        layers_resnet = nn.Sequential(
            OrderedDict(
                [
                    ('dropout1', nn.Dropout(0.5)),
                    ('fc1', nn.Linear(2048, 1024)),
                    ('activation1', nn.ReLU()),
                    ('dropout2', nn.Dropout(0.3)),
                    ('fc2', nn.Linear(1024, 256)),
                    ('activation2', nn.ReLU()),
                    ('dropout3', nn.Dropout(0.3)),
                    ('fc3', nn.Linear(256, 128)),
                    ('activation3', nn.ReLU()),
                    ('fc4', nn.Linear(128, 1))
                ]
            )
        )

        model.fc = layers_resnet
        return get_preds(model, checkpoint, transform, device)


    def agent_seven(device, transform):
        checkpoint = os.path.join(model_dir, "resnet152/checkpoint-best.pth.tar")
        model = models.resnet152(pretrained=False)

        layers_resnet = nn.Sequential(
            OrderedDict(
                [
                    ('dropout1', nn.Dropout(0.5)),
                    ('fc1', nn.Linear(2048, 1024)),
                    ('activation1', nn.ReLU()),
                    ('dropout2', nn.Dropout(0.3)),
                    ('fc2', nn.Linear(1024, 256)),
                    ('activation2', nn.ReLU()),
                    ('dropout3', nn.Dropout(0.3)),
                    ('fc3', nn.Linear(256, 128)),
                    ('activation3', nn.ReLU()),
                    ('fc4', nn.Linear(128, 1))
                ]
            )
        )

        model.fc = layers_resnet
        return get_preds(model, checkpoint, transform, device)


    def agent_eight(device, transform):
        checkpoint = os.path.join(model_dir, "vgg19_bn/checkpoint-best.pth.tar")
        model = models.vgg19_bn(pretrained=False)
        layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('activation1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(4096, 128)),
            ('activation2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(128, 1))
        ]))

        model.classifier = layers
        return get_preds(model, checkpoint, transform, device)
