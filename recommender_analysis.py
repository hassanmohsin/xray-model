import os

import pandas as pd
from sklearn import metrics


def main(agent_name):
    output_dir = "./v3"
    csv_file = os.path.join(output_dir, f"performances/{agent_name}-performance-validation-set.csv")
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"{csv_file} not found.")

    df = pd.read_csv(csv_file, dtype={"image_id": str})

    csv_file = os.path.join(output_dir, f"perf-predictor/resnet50/{agent_name}/{agent_name}-prediction-probability.csv")
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"{csv_file} not found.")

    probs = pd.read_csv(
        csv_file,
        dtype={"ImageId": str}
    )
    df['perf_predictor'] = probs.Label.apply(lambda x: round(x))
    acc = metrics.accuracy_score(df.performance, df.perf_predictor)
    f1 = metrics.f1_score(df.performance, df.perf_predictor)
    f2 = metrics.fbeta_score(df.performance, df.perf_predictor, beta=2.0)
    tn, fp, fn, tp = metrics.confusion_matrix(df.performance, df.perf_predictor).ravel()

    print(f"Accuracy: {acc:.3f}\nF1-score: {f1:0.3f}")
    print(f"F2-score: {f2:0.3f}\nConfusion matrix: \nTN={tn}, FP={fp}\nFN={fn}, TP={tp}")


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

    for agent in agents:
        print("*" * 10, agent.upper(), "*" * 10)
        main(agent)
        print()
