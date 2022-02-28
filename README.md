# xray-model

## Agent

- Train the agents: `python -m agent.train`
  - This will train the agents using the configuration in `agent/config.py` and agent-specific settings in `configs` directory.
- Evaluate the agents on all the datasets: `python -m predictor.evaluate`
  - This will evaluate each agents on the validation set and the test set in each dataset and create prediction files in the corresponding agent directories.

## Predictor training

- Run `python -m predictor.performance` on validation and test set to get the performance evaluation of all the models. `agent-performance-<set>.csv` will be created
- Run `python -m predictor.train` to train the recommender model on the performance results
- Run `python -m predictor.train --input configs/baseline.json` to evaluate only

## Human Experiment

- Run `python -m human_exp.train --input configs/baseline.json` to retrain the predictor models on the human experiment data and get the comparative results for optimized and random assignment.
