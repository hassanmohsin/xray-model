import json
import pandas as pd
from PIL import Image as Im
from itertools import combinations
import random
from sklearn.metrics import accuracy_score

# Find the common participants
def find_common_participants(json_file1, json_file2, json_file3):
    # Read the json files
    with open(json_file1) as f:
        data1 = json.load(f)
    with open(json_file2) as f:
        data2 = json.load(f)
    with open(json_file3) as f:
        data3 = json.load(f)

    participants1 = [response["name"] for response in data1]
    participants2 = [response["name"] for response in data2]
    participants3 = [response["name"] for response in data3]

    # Find the common participants
    common_participants = list(
        set(participants1).intersection(participants2, participants3)
    )
    return common_participants


def read_json(json_files):
    common_participants = find_common_participants(*json_files)
    ground_truth = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ] * 3
    output = {}
    for i, json_file in enumerate(json_files):
        with open(json_file) as f:
            data = json.load(f)

        for response in data:
            name = response["name"]
            if not name in common_participants:
                continue
            entry = {}
            for q in range(1, 31):
                resp = response[str(q)]
                ans = resp["q1"][0]
                dataset_name = resp["dataset"]
                entry[q + 30 * i] = {
                    "pred": ans,
                    "label": ground_truth[q + 30 * i - 1],
                    "dataset": dataset_name,
                    "correct": ans == ground_truth[q + 30 * i - 1],
                }
            if name in output:
                output[name].update(entry)
            else:
                output[name] = entry

    return output


class Mapping:
    def __init__(
        self, json_files: list, number_of_agents: int, type: str = "best"
    ) -> None:
        """
        performance: a list of performance metrics of each agent
        number_of_agents: number of agents in the group
        """
        self.performance = None
        self.json_files = json_files
        self.get_performance()
        self.agents = self.performance.name.values
        self.number_of_agents = number_of_agents
        self.type = type
        self.combinations = list(combinations(self.agents, self.number_of_agents))

    def get_performance(self) -> None:
        self.answers_dict = read_json(self.json_files)
        # Convert the answers to a dataframe
        dfs = []
        for name in self.answers_dict.keys():
            df = pd.DataFrame.from_dict(self.answers_dict[name], orient="index")
            df["name"] = pd.Series(name, index=df.index)
            df["image_id"] = pd.Series(range(1, 91), index=df.index)
            dfs.append(df[["name", "image_id", "pred", "label", "dataset", "correct"]])

        self.answers_df = pd.concat(dfs, ignore_index=True)

        # Calculate the performance of the agents
        self.performance = (
            self.answers_df.groupby("name")
            .apply(
                lambda x: x.groupby("dataset", as_index=False).apply(
                    lambda y: accuracy_score(y["label"], y["pred"])
                )
            )
            .reset_index()
        )
        self.performance["avg"] = self.performance.iloc[:, 1:].mean(axis=1)
        self.performance.sort_values(by="avg", ascending=False, inplace=True)
        self.performance.reset_index(drop=True, inplace=True)

    def get_agents(self) -> list:
        if self.type == "random":
            return self.combinations[random.randint(0, len(self.combinations))]
        elif self.type == "best":
            # chooses the top 4, the middle 1 and the last 1
            agents = self.agents[[0, 1, 2, 3, 7, -1]]
            return agents

        else:
            raise ValueError("Invalid type")


if __name__ == "__main__":
    json_files = [
        "./human_exp/exp1.json",
        "./human_exp/exp2.json",
        "./human_exp/exp3.json",
    ]
    mapping = Mapping(json_files=json_files, number_of_agents=6, type="best")
    print(mapping.get_agents())
