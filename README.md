# metabolite-protein-interactions

This repository contains code aiming to predict metabolite-protein interaction using data from PROMIS experiments and diverse set of features from the metabolite and the protein.

## Project structure
The code is split along several files:
- DataHandler.py: Contains the code necessary for downloading and preprocessing the data.
- datasets.py: contains utility related to the creation of datasets inheriting from torch datasets.
- MPimports.py: contains all the imports of the project.
- PROMISGCEnd2end.py: Contains the simple baseline model, using only features derived from the PROMIS dataset.
- SequenceDLEnd2end.py: Contains the classes derived from torch.nn.Module, and wrappers around functions using for training, testing, and prediction.
- PROMISEnsembleEnd2end.py: Contains the xgboost model and utilities, using a gradient boosting classifier and the predictions from the deep architecture as features.

## Examples
The project contains also several examples of driver code:
- hyperparameter_optimization.ipynb: Contains the code used for doing hyperparameter optimization and comparing the ROC curves of the different modeling approaches.
- metabolite_protein_interactions.ipynb: Contains code assessing the behaviour of the model for pairs consisting of different protein families and metabolites.
