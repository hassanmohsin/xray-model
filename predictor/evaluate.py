import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from agent import AgentConfig, AgentGroup
from agent.dataset import XrayImageDataset
from predictor.dataset import PredictorDataset
from predictor.performance import apply_dropout

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_predictor_dataloaders(agent_group, set, sample_count=None, shuffle=False, pin_memory=True):
    loaders = {}

    for agent in agent_group.agents:
        # Load normalization data
        # TODO: Check if the following is valid
        norm_data_file = os.path.join(agent.params["agent_dir"], "norm_data.pth")
        assert os.path.isfile(norm_data_file)
        norm_data = torch.load(norm_data_file)

        loaders[agent.name] = DataLoader(
            AgentDataset(
                os.path.join(agent.params["performance_dir"], f"{agent.name}-performance-{set}-set.csv"),
                img_dir=os.path.join(agent.params['dataset_dir'], f"{set}-set"),
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


def load_agent_dataloaders(agent_group, set, sample_count=None, shuffle=False, pin_memory=True):
    loaders = {}
    for agent in agent_group.agents:
        # Load normalization data
        norm_data_file = os.path.join(agent.model_dir, "norm_data.pth")
        assert os.path.isfile(norm_data_file)
        norm_data = torch.load(norm_data_file)
        # Create dataset
        loaders[agent.name] = DataLoader(
            XrayImageDataset(
                annotations_file=os.path.join(agent.params["dataset_dir"], f"{set}-labels.csv"),
                img_dir=os.path.join(agent.params["dataset_dir"], f"{set}-set"),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_data["mean"],
                                         std=norm_data["std"])
                ]),
                sample_count=sample_count
            ),
            batch_size=64,  # agent.params["batch_size"],
            shuffle=shuffle,
            num_workers=16,  # agent.params['num_worker'],
            pin_memory=pin_memory
        )

    return loaders


def load_performance_dataloaders(agent_group, set, sample_count=None, shuffle=False, pin_memory=True):
    loaders = {}
    for agent in agent_group:
        # Load normalization data
        norm_data_file = os.path.join(agent.model_dir, "norm_data.pth")
        assert os.path.isfile(norm_data_file)
        norm_data = torch.load(norm_data_file)
        # Create the dataset
        loaders[agent.name] = DataLoader(
            AgentDataset(
                os.path.join(AgentConfig.predictor_dir, f"{agent.name}-performance-{set}-set.csv"),
                img_dir=os.path.join(agent.params['dataset_dir'], f"{set}-set"),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_data["mean"],
                                         std=norm_data["std"])
                ]),
                sample_count=sample_count
            ),
            batch_size=64,  # agent.params['batch_size'],
            shuffle=shuffle,
            num_workers=16,  # agent.params['num_worker'],
            pin_memory=pin_memory
        )

    return dict(zip([agent.name for agent in agent_group.agents], loaders))


def evaluate_agent(agent, dataloader, dataset_name, prediction_count):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = agent.model
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    checkpoint = torch.load(
        os.path.join(agent.model_dir, f"checkpoint-best.pth.tar")
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if prediction_count > 1:
        print(f"Multiple predictions ({prediction_count}) with dropout activated.")
        # apply dropout during inference
        model.apply(apply_dropout)

    indices, labels, predictions = [], [], defaultdict(list)
    with torch.no_grad():
        for idx, image, label in tqdm(dataloader, desc=f"Evaluating {agent.name} on {dataset_name} "):
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


def evaluate(agent_group, assignment, assignment_type="optimized"):
    for agent in agent_group.agents:
        image_ids = assignment[agent.name]
        scores = []
        for dataset_name in [agent.name for agent in agent_group.agents]:
            dataset_image_ids = [_id for _id in image_ids if dataset_name in _id]
            scores.append(pd.read_csv(os.path.join(
                agent.model_dir, f"{agent.name}_on_{dataset_name}-test.csv")
            ).set_index('image_id').loc[dataset_image_ids, 'label'].mean())
        print(sum(scores) / 4)
        break


def main():
    multiple_predictions = 3
    agent_group = AgentGroup(AgentConfig.config_dir)

    # Get the predictions from the agents for all the datasets
    for set_name in ["validation", "test"]:
        dataloaders = load_agent_dataloaders(agent_group, set=set_name, sample_count=None)
        for agent in agent_group.agents:
            dfs = []
            for dataset_name, loader in dataloaders.items():
                output_file = os.path.join(agent.model_dir, f"{agent.name}_on_{dataset_name}-{set_name}.csv")
                if os.path.isfile(output_file):
                    dfs.append(pd.read_csv(output_file, dtype={"label": int}))
                else:
                    df = evaluate_agent(agent, loader, dataset_name, prediction_count=3)
                    df.to_csv(
                        output_file,
                        index=False
                    )
                    dfs.append(df)
            pd.concat(dfs).to_csv(
                os.path.join(agent.model_dir, f"{agent.name}-predictions-on-{set_name}-set.csv"),
                index=False
            )

    # # Get the prediction from the predictors for all the datasets
    # set_name = "test"  # Evaluating only on the test-set
    # dataloaders = load_agent_dataloaders(agent_group, set=set_name, sample_count=1e3)  # TODO: remove sample count
    # for agent in agent_group.agents:
    #     for dataset_name, loader in dataloaders.items():
    #         df = evaluate_predictor(agent, loader, dataset_name, prediction_count=3)
    #         df.to_csv(
    #             os.path.join(agent.params["agent_dir"], f"{agent.name}_on_{dataset_name}-{set_name}.csv"),
    #             index=False
    #         )
    #     # TODO: merge the dfs for each agent and write in one single file

    # Adhoc work using existing data
    probabilities = pd.concat(
        [
            pd.read_csv(
                os.path.join(
                    AgentConfig.predictor_dir,
                    agent.name,
                    f"{agent.name}-prediction-probability{'-multi' if multiple_predictions > 1 else ''}.csv"
                ),
                usecols=['image_id', 'proba_mean'] if multiple_predictions > 1 else ['image_id', 'proba'],
                dtype={'image_id': str}
            ).set_index('image_id') for agent in agent_group.agents
        ],
        axis=1
    ).transpose().dropna(axis=1)
    # TODO: Remove the above dropna operation and investigate what's wrong

    probabilities.to_csv(
        os.path.join(
            AgentConfig.predictor_dir,
            "assignment_probabilities.csv"
        )
    )

    # Find optimized assignment
    assignment = []
    random_assignment = []
    for i in range(0, probabilities.shape[1], len(agent_group.agents)):
        df = probabilities.iloc[:, i:i + len(agent_group.agents)]
        img_ids = df.columns
        _agent_indices = linear_sum_assignment(df.to_numpy(), maximize=True)[1]
        assignment.append(img_ids[_agent_indices].tolist())
        img_ids = img_ids.tolist()
        np.random.shuffle(img_ids)
        random_assignment.append(img_ids)

    # TODO: Take care of the last element that has less than agent_count elements
    assignment = pd.DataFrame.from_records(
        np.stack(assignment[:-1], axis=0),
        columns=[agent.name for agent in agent_group.agents]
    )
    assignment.to_csv(
        os.path.join(AgentConfig.predictor_dir, "assignment-optimal.csv"),
        index=False
    )
    random_assignment = pd.DataFrame.from_records(
        np.stack(random_assignment[:-1], axis=0),
        columns=[agent.name for agent in agent_group.agents]
    )
    random_assignment.to_csv(
        os.path.join(AgentConfig.predictor_dir, "assignment-random.csv"),
        index=False
    )

    print("Evaluating optimized assignment")
    evaluate(agent_group, assignment, assignment_type="optimized")
    print("Evaluating random assignment")
    evaluate(agent_group, random_assignment, assignment_type="random")


if __name__ == '__main__':
    main()
