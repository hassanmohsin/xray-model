import json
import os
from argparse import ArgumentParser
from functools import partial
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import AgentGroup
from agent.config import AgentConfig
from agent.models import get_model
from predictor.dataset import PredictorDataset
from utils.utils import get_mean_std, save_checkpoint, load_checkpoint, binary_acc, apply_dropout

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

seed = 42
# torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def train(args, agent, evaluate=False, resume_training=False, multiple_predictions=3):
    # args is for predictor config, agent is what the predictor is for.
    model_dir = os.path.join(
        AgentConfig.predictor_dir, agent.name, args['model_name']
    )
    writer = SummaryWriter(model_dir)

    def send_stats(i, module, input, output):
        writer.add_scalar(f"{i}-mean", output.data.std())
        writer.add_scalar(f"{i}-stddev", output.data.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find the mean and std of training data using a subset of training data
    norm_file = os.path.join(model_dir, "norm_data_predictor.pth")
    if os.path.isfile(norm_file):
        norm_data = torch.load(norm_file)
        mean, std = norm_data['mean'], norm_data['std']
    else:
        sample_size = 100_000
        print(f"Calculating mean and std of the training set on {sample_size} samples.")
        validation_data = PredictorDataset(
            agent,
            subset="validation",
            transform=transforms.ToTensor(),
            sample_count=sample_size
        )

        validation_loader = DataLoader(
            validation_data,
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args["num_workers"],
            pin_memory=True
        )

        # use the mean and std to normalize
        mean, std = get_mean_std(validation_loader)
        torch.save(
            {
                'mean': mean,
                'std': std
            },
            norm_file
        )
    print(f"Training data : mean: {mean} and std: {std}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ]
    )

    datasets = {
        'validation':
            PredictorDataset(
                agent,
                transform=transform,
                subset='validation',
                # sample_count=10_000
            ),
        'test':
            PredictorDataset(
                agent,
                transform=transform,
                subset='test',
                # sample_count=10_000
            )
    }

    loaders = {
        'validation':
            DataLoader(
                datasets['validation'],
                batch_size=args["batch_size"],
                shuffle=True,
                num_workers=args["num_workers"],
                pin_memory=True
            ),
        'test':
            DataLoader(
                datasets['test'],
                batch_size=args["batch_size"],
                shuffle=True,
                num_workers=args["num_workers"],
                pin_memory=True
            )
    }

    model = get_model(args['model_name'])

    if evaluate:
        print("Evaluating on the test set...")
        checkpoint = torch.load(
            os.path.join(model_dir, f"checkpoint-best.pth.tar")
        )
        model.load_state_dict(checkpoint['state_dict'])
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)
        model.eval()

        if multiple_predictions:
            print(f"Multiple predictions ({multiple_predictions}) with dropout activated.")
            # apply dropout during inference
            model.apply(apply_dropout)

            def predict_agent_dataset(loader):
                print(f"{args['model_name']} predicting the test set...")
                # TODO: Don't hardcode number of predictions
                # TODO: Remove duplicate code segment below
                # TODO: Move this loop to where the model is spitting out the predictions.
                #  Make sure the output is not identical
                indices, labels, predictions = [], [], defaultdict(list)
                with torch.no_grad():
                    for idx, image, label in tqdm(loader, desc=f"Evaluating {args['model_name']} on test set"):
                        inputs = image.to(device)
                        labels += label.numpy().tolist()
                        preds = [torch.sigmoid(model(inputs)).squeeze().cpu().numpy() for _ in range(multiple_predictions)]
                        for i, pred in enumerate(preds):
                            predictions[f"proba_{i}"] += pred.tolist()
                        indices += idx

                pred_df = pd.DataFrame()
                pred_df["image_id"] = pd.Series(indices)
                pred_df["label"] = pd.Series(labels)
                for k, v in predictions.items():
                    pred_df[k] = pd.Series(v)
                pred_df["proba_mean"] = pred_df.mean(axis=1, numeric_only=True)
                pred_df["proba_var"] = pred_df.var(axis=1, numeric_only=True)

                return pred_df

            predict_agent_dataset(loaders['test']).to_csv(
                os.path.join(model_dir, f"{agent.name}-prediction-probability-test-set.csv"),
                index=False
            )

        else:
            indices = []
            predictions = []
            with torch.no_grad():
                for idx, image, label in tqdm(loaders['test'], desc=f"Evaluating {args['model_name']} on test set"):
                    inputs = image.to(device)
                    preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
                    predictions += preds.tolist()
                    indices += idx

            pd.DataFrame(
                {
                    'image_id': indices,
                    'proba': predictions
                }
            ).to_csv(
                os.path.join(model_dir, f'{agent.name}-prediction-probability-test-set.csv'),
                index=False
            )

        return

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # writer.add_graph(model, torch.rand([1, 3, 224, 224]))
    start_epoch = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=agent.params['learning_rate'])
    if resume_training:
        model, optimizer, start_epoch = load_checkpoint(
            model,
            optimizer,
            os.path.join(model_dir, "checkpoint-best.pth.tar")
        )

    best_accuracy = 0.0
    for epoch in range(start_epoch, args['epochs'] + 1):  # loop over the dataset multiple times
        epoch_loss, epoch_acc = 0., 0.
        running_loss, running_acc = 0., 0.
        model.train()
        for i, (image_ids, images, labels) in enumerate(loaders['validation'], 0):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            # add to tensorboard
            # grid = torchvision.utils.make_grid(images[:8])
            # writer.add_image('train/images', grid, epoch)
            # writer.add_graph(model, inputs)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = binary_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            running_loss += loss.item()
            running_acc += acc.item()
            if i % 10 == 0:
                print(f'iter: {i:04}: Running loss: {running_loss / 10:.3f} | Running acc: {running_acc / 10:.3f}')
                running_loss = 0.0
                running_acc = 0.0

        model.eval()
        val_accuracy = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for i, (image_ids, images, labels) in enumerate(loaders['test'], 0):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                # add to tensorboard
                # grid = torchvision.utils.make_grid(images[:8])
                # writer.add_image('validation/images', grid, epoch)

                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = binary_acc(outputs, labels)
                val_accuracy += acc.item()
                val_loss += loss.item()

        val_accuracy = val_accuracy / len(loaders['test'])
        val_loss = val_loss / len(loaders['test'])
        is_best = bool(val_accuracy > best_accuracy)
        best_accuracy = max(val_accuracy, best_accuracy)
        # Save checkpoint if is a new best
        save_checkpoint(
            {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.module.state_dict(),
                'loss': val_loss,
                'best_accuracy': best_accuracy
            },
            is_best,
            filename=os.path.join(model_dir, f'checkpoint-{epoch:03d}-val-{val_accuracy:.3f}.pth.tar')
        )

        train_loss = epoch_loss / len(loaders['validation'])
        train_acc = epoch_acc / len(loaders['validation'])
        print(
            f'Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc:'
            f' {train_acc:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_accuracy:.3f} | Best Acc: {best_accuracy:.3f}'
        )

        # add to tensorboard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/accuracy", val_accuracy, epoch)

        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight, epoch)

        for i, m in enumerate(model.children()):
            m.register_forward_hook(partial(send_stats, i))

        writer.flush()

    print('Finished Training.')


#
# # TODO: Remove `args` argument from the following method, these should be available in agent.params
# def train_agent(agent, args, evaluate_only=False, multiple_predictions=False):
#     model_dir = os.path.join(AgentConfig.predictor_dir, agent.name)
#     if not os.path.isdir(model_dir):
#         os.makedirs(model_dir)
#     writer = SummaryWriter(os.path.join(model_dir, 'logs'))
#     batch_size = args['batch_size']
#     epochs = 3
#     learning_rate = args['learning_rate']
#     num_workers = args['num_worker']
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
#     dataset = AgentDataset(
#         agent.name,
#         os.path.join(agent.model_dir, f"{agent.name}-predictions-on-validation-set.csv"),
#         img_dir=os.path.join(agent.params['dataset_dir'], "validation-set"),
#         transform=transform,
#         sample_count=1000
#     )
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#
#     validation_set = AgentDataset(
#         agent.name,
#         os.path.join(agent.model_dir, f"{agent.name}-predictions-on-test-set.csv"),
#         img_dir=os.path.join(agent.params['dataset_dir'], "test-set"),
#         transform=transform,
#         sample_count=1000
#     )
#
#     validation_loader = DataLoader(
#         validation_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#
#     model = None
#     if args["model_name"] in ["resnet34", "resnet18"]:
#         model = models.resnet34(
#             pretrained=args["pretrained"]
#         ) if args["model_name"] == "resnet34" else models.resnet18(
#             pretrained=args["pretrained"]
#         )
#
#         if args['pretrained']:
#             for name, param in model.named_parameters():
#                 if 'bn' not in name:  # DON'T freeze BN layers
#                     param.requires_grad = False
#
#         model.fc = nn.Sequential(OrderedDict([
#             ('dropout1', nn.Dropout(0.5)),
#             ('fc1', nn.Linear(512, 256)),
#             ('activation1', nn.ReLU()),
#             ('dropout2', nn.Dropout(0.3)),
#             ('fc2', nn.Linear(256, 128)),
#             ('activation2', nn.ReLU()),
#             ('fc3', nn.Linear(128, 1))
#         ]))
#
#     elif args["model_name"] in ["resnet50", "resnet101", "resnet152", "wide_resnet101_2"]:
#         if args["model_name"] == "resnet50":
#             model = models.resnet50(pretrained=args['pretrained'])
#         elif args["model_name"] == "resnet101":
#             model = models.resnet101(pretrained=args['pretrained'])
#         elif args["model_name"] == "resnet152":
#             model = models.resnet152(pretrained=args['pretrained'])
#         elif args["model_name"] == "wide_resnet101_2":
#             model = models.wide_resnet101_2(pretrained=args['pretrained'])
#
#         if args['pretrained']:
#             for name, param in model.named_parameters():
#                 if 'bn' not in name:  # DON'T freeze BN layers
#                     param.requires_grad = False
#
#         model.fc = nn.Sequential(
#             OrderedDict(
#                 [
#                     ('dropout1', nn.Dropout(0.5)),
#                     ('fc1', nn.Linear(2048, 1024)),
#                     ('activation1', nn.ReLU()),
#                     ('dropout2', nn.Dropout(0.3)),
#                     ('fc2', nn.Linear(1024, 256)),
#                     ('activation2', nn.ReLU()),
#                     ('dropout3', nn.Dropout(0.3)),
#                     ('fc3', nn.Linear(256, 128)),
#                     ('activation3', nn.ReLU()),
#                     ('fc4', nn.Linear(128, 1))
#                 ]
#             )
#         )
#     elif args["model_name"] == "vgg19_bn":
#         model = models.vgg19_bn(pretrained=args["pretrained"])
#
#         if args['pretrained']:
#             for name, param in model.named_parameters():
#                 if 'bn' not in name:  # DON'T freeze BN layers
#                     param.requires_grad = False
#
#         model.classifier = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(25088, 4096)),
#             ('activation1', nn.ReLU()),
#             ('dropout1', nn.Dropout(0.5)),
#             ('fc2', nn.Linear(4096, 128)),
#             ('activation2', nn.ReLU()),
#             ('dropout2', nn.Dropout(0.3)),
#             ('fc3', nn.Linear(128, 1))
#             # ('out', nn.Sigmoid())
#         ]))
#
#     else:
#         raise NotImplementedError("Model not found")
#
#     assert (model is not None)
#
#     if torch.cuda.device_count() > 1:
#         print("Using ", torch.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model)
#
#     model.to(device)
#
#     if evaluate_only:
#         print("Evaluating on the test set...")
#         checkpoint = torch.load(
#             os.path.join(model_dir, f"checkpoint-best.pth.tar")
#         )
#         model.load_state_dict(checkpoint['state_dict'])
#         model.eval()
#
#         if multiple_predictions:
#             prediction_count = 3
#             print(f"Multiple predictions ({prediction_count}) with dropout activated.")
#             # apply dropout during inference
#             model.apply(apply_dropout)
#
#             def predict_agent_dataset(dataset_name, loader):
#                 print(f"Predicting on {dataset_name}...")
#                 # TODO: Don't hardcode number of predictions
#                 # TODO: Remove duplicate code segment below
#                 dfs = []
#                 # TODO: Move this loop to where the model is spitting out the predictions.
#                 #  Make sure the output is not identical
#                 for i in range(1, prediction_count + 1):
#                     indices = []
#                     predictions = []
#                     with torch.no_grad():
#                         for idx, image, label in tqdm(loader, desc=f"Evaluating {agent.name}"):
#                             inputs = image.to(device)
#                             preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
#                             predictions += preds.tolist()
#                             indices += idx
#                     dfs.append(
#                         pd.DataFrame(
#                             {
#                                 "image_id": [dataset_name + '-' + i for i in indices],
#                                 f"proba_{i}": predictions
#                             },
#                         ).set_index('image_id')
#                     )
#
#                 all_dfs = pd.concat(dfs, axis=1)
#                 # all_dfs['image_id'] = all_dfs['image_id'].apply(
#                 #     lambda x: dataset_name + '-' + x,
#                 # )
#                 # all_dfs.set_index('image_id')
#                 all_dfs['proba_mean'] = all_dfs.mean(axis=1, numeric_only=True)
#                 all_dfs['proba_var'] = all_dfs.var(axis=1, numeric_only=True)
#                 return all_dfs
#
#             agent_group = AgentGroup(AgentConfig.config_dir)
#             dataloaders = load_all_datasets(
#                 agent_group,transform
#             )
#
#             agent_prediction_dfs = []
#             for name, dataloader in dataloaders.items():
#                 agent_prediction_dfs.append(predict_agent_dataset(name, dataloader))
#
#             pd.concat(agent_prediction_dfs, axis=0).to_csv(
#                 os.path.join(model_dir, f"{agent.name}-prediction-probability-multi.csv")
#             )
#
#         else:
#             indices = []
#             predictions = []
#             with torch.no_grad():
#                 for idx, image, label in tqdm(validation_loader, desc=f"Evaluating {agent.name}"):
#                     inputs = image.to(device)
#                     preds = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()
#                     predictions += preds.tolist()
#                     indices += idx
#
#             pd.DataFrame(
#                 {
#                     'image_id': indices,
#                     'proba': predictions
#                 }
#             ).to_csv(
#                 os.path.join(model_dir, f'{agent.name}-prediction-probability.csv'),
#                 index=False
#             )
#
#         return
#
#     # criterion = nn.BCEWithLogitsLoss()
#     criterion = FocalLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()
#
#     best_acc = 0.0
#     for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
#         model.train()
#         epoch_loss = 0.0
#         epoch_acc = 0.0
#         running_loss = 0.0
#         running_acc = 0.0
#         for i, (idx, x, y) in enumerate(data_loader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = x.to(device), y.to(device).unsqueeze(1)
#             labels = labels.to(torch.float)  # For BCEwithLogits loss
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             acc = binary_acc(outputs, labels)
#
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
#             running_loss += loss.item()
#             running_acc += acc.item()
#             if i % 10 == 0:
#                 print(f'iter: {i:04}: Running loss: {running_loss / 10:.3f} | Running acc: {running_acc / 10:.3f}')
#                 running_loss = 0.0
#                 running_acc = 0.0
#
#         train_loss = epoch_loss / len(data_loader)
#         train_acc = epoch_acc / len(data_loader)
#         print(f'Epoch {epoch:03}: Loss: {train_loss:.3f} | Acc: {train_acc:.3f}')
#         is_best = bool(train_acc > best_acc)
#         best_acc = max(best_acc, train_acc)
#         save_checkpoint(
#             {
#                 'epoch': epoch,
#                 'optimizer': optimizer.state_dict(),
#                 'state_dict': model.state_dict(),
#                 'loss': train_loss,
#                 'best_acc': best_acc
#             },
#             is_best,
#             filename=os.path.join(model_dir, f'checkpoint-{epoch:03d}-val-{train_acc:.3f}.pth.tar')
#         )
#
#         # add to tensorboard
#         writer.add_scalar("Loss/train", train_loss, epoch)
#         writer.add_scalar("Acc/train", train_acc, epoch)
#         # writer.add_scalar("Loss/validation", val_loss, epoch)
#
#         writer.flush()
#
#     print('Finished Training.')


def evaluate(args, agent_group, assignment, assignment_type="optimized"):
    ind_eval = {}
    evaluation = {}
    total_acc, total_fbeta = 0., 0.
    for agent in agent_group.agents:
        # Read the test performance file
        df = pd.read_csv(
            os.path.join(AgentConfig.predictor_dir, agent.name, args['model_name'], f"{agent.name}-prediction-probability-test-set.csv"),
            dtype={"image_id": str}
        )
        df['pred_label'] = df.proba_mean.apply(lambda x: round(x))
        df['success'] = df.apply(lambda row: int(row['label'] == row['pred_label']), axis=1)

        # individual performance
        acc = df.success.mean()
        fbeta = metrics.fbeta_score(
            df.label.to_list(),
            df.pred_label.to_list(),
            beta=2.0
        )
        ind_eval[agent.name] = {
            "accuracy": f"{acc:.3f}",
            "fbeta": f"{fbeta:.3f}"
        }
        # assigned performance
        _df = df[df.image_id.isin(assignment[agent.name])]
        acc = _df.success.mean()
        total_acc += acc
        fbeta = metrics.fbeta_score(_df.label.to_list(), _df.pred_label.to_list(), beta=2.0)
        total_fbeta += fbeta
        evaluation[agent.name] = {
            "accuracy": f"{acc:.3f}",
            "fbeta": f"{fbeta:.3f}"
        }

    evaluation['avg_acc'] = f"{total_acc / len(agent_group.agents):.3f}"
    evaluation['avg_fbeta'] = f"{total_fbeta / len(agent_group.agents):.3f}"
    final_eval = {
        'individual': ind_eval,
        'assigned': evaluation
    }
    print(final_eval)
    with open(os.path.join(
            AgentConfig.predictor_dir,
            # TODO: Remove hardcoded index, get params in a generalized way
            f"./{assignment_type}-assignment-performance.json"
    ), 'w') as f:
        json.dump(final_eval, f)


def main(args, training=True):
    agent_group = AgentGroup()
    # Train and evaluate to get the probabilities
    multiple_prediction = 3
    for agent in agent_group.agents:
        print(f"Training and evaluating {agent.name}")
        if training:
            train(args, agent, evaluate=False)
        train(args, agent, evaluate=True, multiple_predictions=multiple_prediction)

    # Probability matrix need to be in the following format:
    #         image_1 image_2 image_3 ... image_n
    # agent_1
    # agent_2
    # agent_3
    # ...
    # ...
    # agent_n
    #
    # TODO: Get the probabilities for all the datasets (we have agent-specific datasets)
    #  while doing so, normalize the test sets using dataset specific mean and std
    probabilities = pd.concat(
        [
            pd.read_csv(
                os.path.join(
                    AgentConfig.predictor_dir,
                    agent.name,
                    args['model_name'],
                    f"{agent.name}-prediction-probability-test-set.csv"
                ),
                usecols=['image_id', 'proba_mean'] if multiple_prediction else ['image_id', 'proba'],
                dtype={'image_id': str}
            ).set_index('image_id') for agent in agent_group.agents
        ],
        axis=1
    ).transpose().dropna(axis=1)  # Transpose is to get the matrix in the required format
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
        os.path.join(AgentConfig.predictor_dir, "assignment.csv"),
        index=False
    )
    random_assignment = pd.DataFrame.from_records(
        np.stack(random_assignment[:-1], axis=0),
        columns=[agent.name for agent in agent_group.agents]
    )

    print("Evaluating optimized assignment")
    evaluate(args, agent_group, assignment, assignment_type="optimized")
    print("Evaluating random assignment")
    evaluate(args, agent_group, random_assignment, assignment_type="random")


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a recommender model')
    parser.add_argument('--input', type=str, required=True, action='store',
                        help="JSON input")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate only")
    args_cmd = parser.parse_args()
    if not os.path.isfile(args_cmd.input):
        raise FileNotFoundError(f"Input {args_cmd.input} not found.")

    with open(args_cmd.input) as f:
        args = json.load(f)

    if args_cmd.evaluate:
        main(args, training=False)
    else:
        main(args, training=True)
