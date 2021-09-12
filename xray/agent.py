import json
import os
from collections import OrderedDict

from torch import nn
from torchvision import models

from xray.config import AgentConfig


class Agent(object):
    def __init__(self, agent_name, params_file):
        self.name = agent_name
        with open(params_file) as f:
            self.params = json.load(f)
        self.model = None
        self.params['agent_dir'] = os.path.join(
            self.params.get("output_dir", "results"),
            AgentConfig.agent_dir,
            self.params.get("model_name", "model")
        )
        self.params['performance_dir'] = os.path.join(
            self.params.get("output_dir", "results"),
            AgentConfig.parformance_dir
        )

    def set_model(self, model):
        self.model = model

    def __str__(self):
        return f"agent_name: {self.name}"


class AgentGroup(object):
    def __init__(self, param_dir):
        self.agent_names = [
            "agent_one",
            "agent_two",
            "agent_three",
            "agent_four",
            "agent_five",
            "agent_six"
        ]

        self.param_files = [
            "resnet18.json",
            "resnet34.json",
            "resnet50.json",
            "resnet101.json",
            "resnet152.json",
            "wide_resnet101_2.json"
        ]

        self.param_dir = param_dir

        if not os.path.isdir(self.param_dir):
            raise NotADirectoryError(f"{self.param_dir} not found!")

        self.agents = [Agent(name, os.path.join(self.param_dir, param)) for name, param in
                       zip(self.agent_names, self.param_files)]

        self.assign_models()

    def assign_models(self):
        for agent in self.agents:
            if agent.name == "agent_one":
                model = models.resnet18(pretrained=False)
                model.fc = nn.Sequential(OrderedDict([
                    ('dropout1', nn.Dropout(0.5)),
                    ('fc1', nn.Linear(512, 256)),
                    ('activation1', nn.ReLU()),
                    ('dropout2', nn.Dropout(0.3)),
                    ('fc2', nn.Linear(256, 128)),
                    ('activation2', nn.ReLU()),
                    ('fc3', nn.Linear(128, 1))
                    # ('out', nn.Sigmoid())
                ]))
                agent.set_model(model)

            elif agent.name == "agent_two":
                model = models.resnet34(pretrained=False)
                model.fc = nn.Sequential(OrderedDict([
                    ('dropout1', nn.Dropout(0.5)),
                    ('fc1', nn.Linear(512, 256)),
                    ('activation1', nn.ReLU()),
                    ('dropout2', nn.Dropout(0.3)),
                    ('fc2', nn.Linear(256, 128)),
                    ('activation2', nn.ReLU()),
                    ('fc3', nn.Linear(128, 1))
                    # ('out', nn.Sigmoid())
                ]))
                agent.set_model(model)

            elif agent.name == "agent_three":
                model = models.resnet50(pretrained=False)
                model.fc = nn.Sequential(
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
                agent.set_model(model)

            elif agent.name == "agent_four":
                model = models.resnet101(pretrained=False)
                model.fc = nn.Sequential(
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
                agent.set_model(model)

            elif agent.name == "agent_five":
                model = models.resnet152(pretrained=False)
                model.fc = nn.Sequential(
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
                agent.set_model(model)

            elif agent.name == "agent_six":
                model = models.wide_resnet101_2(pretrained=False)
                model.fc = nn.Sequential(
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
                agent.set_model(model)
            else:
                raise NotImplementedError("Not implemented yet")
