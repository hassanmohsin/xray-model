import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from recommender.dataset import XrayImageDataset
from xray.models import ModelTwo

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# TODO: specify dataset type e.g., 'validation' or 'test'
def get_preds(agent_name, model, checkpoint, transform, device, dataset_type):
    # Create the output directory
    output_dir = "performances"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{agent_name}-performance-{dataset_type}-set.csv")

    # skip if already done
    if os.path.isfile(output_file):
        return

    # TODO: get predictions from all models concurrently
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = XrayImageDataset(
        annotations_file=f"{dataset_dir}/{dataset_type}-labels.csv",
        img_dir=f"{dataset_dir}/{dataset_type}-set",
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    ground_truth, predictions = [], []
    success_labels = []
    image_ids = []
    with torch.no_grad():
        for image_id, image, labels in tqdm(dataloader, desc=f"{agent_name} on {dataset_type} set: "):
            pred = model(image.to(device))
            pred = torch.round(torch.sigmoid(pred)).squeeze().cpu()
            ground_truth += labels.to(torch.int).tolist()
            predictions += pred.to(torch.int).tolist()
            success_labels += (pred == labels).to(torch.int).tolist()
            image_ids += image_id

    pd.DataFrame.from_dict(
        {
            'image_id': image_ids,
            'label': ground_truth,
            'prediction': predictions,
            'performance': success_labels
        }
    ).to_csv(
        output_file,
        index=False
    )


def agent_one(device, transform, dataset_type):
    checkpoint = os.path.join(model_dir, "ModelTwo/checkpoint-best.pth.tar")
    model = ModelTwo()
    return get_preds('agent_one', model, checkpoint, transform, device, dataset_type)


def agent_two(device, transform, dataset_type):
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
    return get_preds('agent_two', model, checkpoint, transform, device, dataset_type)


def agent_three(device, transform, dataset_type):
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
    return get_preds('agent_three', model, checkpoint, transform, device, dataset_type)


def agent_four(device, transform, dataset_type):
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
    return get_preds('agent_four', model, checkpoint, transform, device, dataset_type)


def agent_five(device, transform, dataset_type):
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
    return get_preds('agent_five', model, checkpoint, transform, device, dataset_type)


def agent_six(device, transform, dataset_type):
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
    return get_preds('agent_six', model, checkpoint, transform, device, dataset_type)


def agent_seven(device, transform, dataset_type):
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
    return get_preds('agent_seven', model, checkpoint, transform, device, dataset_type)


def agent_eight(device, transform, dataset_type):
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
    return get_preds('agent_eight', model, checkpoint, transform, device, dataset_type)


def merge_dfs(dir="performance", dataset_type="validation"):
    validation_df = pd.read_csv(f"{dir}/{agents[0]}-performance-{dataset_type}-set.csv", dtype={"image_id": str})
    for agent in agents[1:]:
        validation_df = pd.merge(
            validation_df,
            pd.read_csv(f"{dir}/{agent}-performance-{dataset_type}-set.csv", dtype={"image_id": str}),
            on="image_id",
            how="inner"
        )

    validation_df.to_csv(f"{dir}/agent-performance-{dataset_type}-set.csv", index=False)


if __name__ == '__main__':
    agents = [
        "agent_one",
        "agent_two",
        "agent_three",
        "agent_four",
        "agent_five",
        "agent_six",
        "agent_seven",
        "agent_eight"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dataset_dir = "/data2/mhassan/xray-dataset/v3"
    model_dir = "/data2/mhassan/xray-model/v3"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.7165, 0.7446, 0.7119],
            [0.3062, 0.2433, 0.2729]
        )
    ])

    for dataset_type in ['validation', 'test']:
        agents_performances = [
            agent_one(device, transform, dataset_type),
            agent_two(device, transform, dataset_type),
            agent_three(device, transform, dataset_type),
            agent_four(device, transform, dataset_type),
            agent_five(device, transform, dataset_type),
            agent_six(device, transform, dataset_type),
            agent_seven(device, transform, dataset_type),
            agent_eight(device, transform, dataset_type)
        ]
        print("Merging dataframes...")
        merge_dfs('./performances', dataset_type)

    # agent_one_img_ids, agent_one_probs = agent_one(device, transform)
    # agent_two_img_ids, agent_two_probs = agent_two(device, transform)
    # agent_three_img_ids, agent_three_probs = agent_three(device, transform)
    # print(agent_one_img_ids, agent_one_probs)

    # Check the order of the images
    # for (i, j) in zip(agent_one_img_ids, agent_two_img_ids):  # , agent_three_img_ids):
    #     assert i == j, "Image indices are not in order."

    # pd.DataFrame.from_dict(
    #     {
    #         'image_id': agents_performances[0][0],
    #         'agent_one': agents_performances[0][1],
    #         'agent_two': agents_performances[1][1],
    #         'agent_three': agents_performances[2][1],
    #         'agent_four': agents_performances[3][1],
    #         'agent_five': agents_performances[4][1]
    #     }
    # ).to_csv(
    #     "agent-performance-validation.csv",
    #     index=False
    # )
