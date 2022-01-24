import xgboost as xgb
import pandas as pd
import numpy as np
import torch
import re
import os
import pickle
from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from utils import *
from datasets import *
from End2EndModel import *
from PROMISGCEnd2end import *
from DataHandler import *


class PROMISEnsembleEnd2end(PROMISGCEnd2end):
    def __init__(self, hyperparameters=None, model_name="PROMISGBCEnsemble", root="./saved_models"):
        self.predictions_dl_path = "matrix_predictions.csv"
        self.gbc_path_e = "gbc_ensemble.pkl"
        super().__init__(hyperparameters, model_name, root)

    def _download_files(self, force=False):
        self.file_emm = os.path.join(self.root, self.gmm_path)
        if os.path.exists(self.file_emm) and not force:
            pass
        else:
            file_id = "1C9kHVLSVad5UBFHIBDYhvdxIqmYUi-cP"
            print("Downloading trained gaussian mixture model")
            download_file_from_google_drive(file_id, self.file_emm)

        self.file_gbc = os.path.join(self.root, self.gbc_path_e)
        if os.path.exists(self.file_gbc) and not force:
            pass
        else:
            file_id = "1a-sLM1V5VSQ2Qo-ovDR6FRciiDdUThgl"
            print("Downloading trained gradient boosting classifier")
            download_file_from_google_drive(file_id, self.file_gbc)

        self.predictions_dl_file = os.path.join(
            self.root, self.predictions_dl_path)
        if os.path.exists(self.predictions_dl_file) and not force:
            pass
        else:
            file_id = "1o3Q2eitbbjrBSPlSU2QMlsgqoqL9Gc2G"
            print("Downloading predictions generated by deep architecture")
            download_file_from_google_drive(file_id, self.predictions_dl_file)

    def preprocess_data(self, threshold=0.6, add_negative_sample=True, ratio_neg=1):
        if os.path.exists(self.goldstandard_path):
            df_interactions = pd.read_csv(
                self.goldstandard_path, index_col=0).reset_index(drop=True)
        else:
            df_interactions = self._generate_gold_standard(threshold)
        tanimoto_global = pd.read_csv(
            "./data/tanimoto_global.csv", index_col=0)
        tanimoto_accessions = pd.read_csv(
            "./data/tanimoto_accessions.csv", index_col=0)
        with open(self.file_emm, "rb") as f:
            gaussian_mixture = pickle.load(f)
        locations_mets = pd.read_csv(
            "./data/metabolite_locations.csv", sep=",")
        locations_prots = pd.read_csv("./data/proteins_locations.csv", sep=",")
        if add_negative_sample:
            df_interactions = self._add_negative_sample(
                df_interactions, df_interactions, ratio_neg)
        df_predictions = pd.read_csv(self.predictions_dl_file, index_col=0)
        df_predictions = df_predictions.set_index(["chemical", "protein"])
        dataset = PromisDatasetXGB(df_interactions,
                                   gaussian_mixture,
                                   tanimoto_accessions, tanimoto_global,
                                   locations_mets,
                                   locations_prots,
                                   df_predictions)
        X, y, mets, prots = dataset.get_data()
        self.data = X
        self.labels = y
        self.data_proteins = prots
        self.data_mets = mets

    def _process_external_data(self, df_interactions):
        tanimoto_global = pd.read_csv(
            "./data/tanimoto_global.csv", index_col=0)
        tanimoto_accessions = pd.read_csv(
            "./data/tanimoto_accessions.csv", index_col=0)
        with open(self.file_emm, "rb") as f:
            gaussian_mixture = pickle.load(f)
        locations_mets = pd.read_csv(
            "./data/metabolite_locations.csv", sep=",")
        locations_prots = pd.read_csv("./data/proteins_locations.csv", sep=",")
        df_predictions = pd.read_csv(self.predictions_dl_file, index_col=0)
        df_predictions = df_predictions.set_index(["chemical", "protein"])
        dataset = PromisDatasetXGB(df_interactions,
                                   gaussian_mixture,
                                   tanimoto_accessions, tanimoto_global,
                                   locations_mets,
                                   locations_prots,
                                   df_predictions)
        X, _, _, _ = dataset.get_data()
        return X