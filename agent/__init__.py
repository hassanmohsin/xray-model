import json
import os

from agent.config import AgentConfig
from agent.models import get_model


class Agent(object):
    def __init__(self, agent_name, params_file):
        self.name = agent_name
        with open(params_file) as f:
            self.params = json.load(f)
        self.model = get_model(
            self.params["model_name"], pretrained=self.params["pretrained"]
        )
        self.model_dir = os.path.join(AgentConfig.agent_dir, self.params["model_name"])

    def __str__(self):
        return f"agent_name: {self.name}"


class AgentGroup(object):
    def __init__(self):
        self.agent_names = AgentConfig.agent_names
        self.param_files = AgentConfig.param_files
        self.param_dir = AgentConfig.config_dir

        if not os.path.isdir(self.param_dir):
            raise NotADirectoryError(f"{self.param_dir} not found!")

        self.agents = [
            Agent(name, os.path.join(self.param_dir, param))
            for name, param in zip(self.agent_names, self.param_files)
        ]

    def get_agent(self, agent_name):
        return dict(zip(self.agent_names, self.agents))[agent_name]
