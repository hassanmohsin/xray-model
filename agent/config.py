class AgentConfig(object):
    agent_names = [
        "agent_one",
        "agent_two",
        "agent_three",
        "agent_four",
        # "agent_five",
        # "agent_six"
    ]
    param_files = [
        "baseline.json",
        "resnet18.json",
        "resnet34.json",
        "resnet50.json",
        # "resnet101.json",
        # "resnet152.json",
        # "wide_resnet101_2.json"
    ]
    parent_dir = "./results"
    agent_dir = "./results/agents"
    predictor_dir = "./results/predictors"
    config_dir = "./configs"