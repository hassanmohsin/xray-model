import os

import pandas as pd

from xray.agent import AgentGroup
from xray.config import AgentConfig

if __name__ == '__main__':
    agent_group = AgentGroup(AgentConfig.config_dir)

    for set_name in ["validation", "test"]:
        for dataset in agent_group.agents:
            gt_file = os.path.join(dataset.params["dataset_dir"], f"{set_name}-labels.csv")
            gt_df = pd.read_csv(gt_file, dtype={"image_id": str})
            gt_df['image_id'] = gt_df.image_id.copy().apply(lambda x: dataset.name + '-' + x)

            for agent in agent_group.agents:
                df = pd.read_csv(
                    os.path.join(
                        agent.params["agent_dir"], f"{agent.name}_on_{dataset.name}-{set_name}.csv"
                    )
                )

                pd.merge(df, gt_df, on='image_id').to_csv(
                    os.path.join(
                        agent.params["agent_dir"], f"{agent.name}_on_{dataset.name}-{set_name}_with_label.csv"
                    ),
                    index=False
                )

# if __name__ == "__main__":
#     data_dir = "./"
#     agent_group = AgentGroup("./configs")
#
#     for agent in agent_group.agents:
#         for set_name in ['validation', 'test']:
#             prediction_files = glob(
#                 os.path.join(agent.params['performance_dir'], '*', f"{agent.name}-performance-{set_name}-set.csv")
#             )
#             df_merged = reduce(
#                 lambda left, right: pd.merge(left, right, on="image_id", how="outer"),
#                 [pd.read_csv(f, dtype={"image_id": str}) for f in prediction_files]
#             )
#             df_merged.set_index("image_id", inplace=True)
#             print(df_merged.head())
#             # print(df_merged.mean(axis=1).head())
#             break
