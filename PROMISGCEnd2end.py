import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle
from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import KFold
from utils import *
from datasets import *
from End2EndModel import *


class PROMISGCEnd2end(End2endModel):
    """
    Interface for prediction metabolite-protein interactions based exclusively on
    the PROMIS data.
    """

    def __init__(self, hyperparameters=None, model_name="PROMISGBC", root="./saved_models"):
        super().__init__(model_name, root)
        if hyperparameters is None:  # default initialization
            self.hyperparameters = PROMISHyperParameters()
        elif type(hyperparameters) == PROMISHyperParameters:
            self.hyperparameters = hyperparameters
        else:
            print("Hyperparameters type must be PROMISHyperParameters")
            raise TypeError
        # paths for the pretrained models
        self.gmm_path = "gmm_trained.pkl"
        self.gbc_path = "gbc_simple.pkl"
        self._download_files()

    def _download_files(self, force=False):
        """
        Downloads the pretrained GBC and GMM if they are not found.
        """
        self.file_emm = os.path.join(self.root, self.gmm_path)
        if os.path.exists(self.file_emm) and not force:
            pass
        else:
            file_id = "1C9kHVLSVad5UBFHIBDYhvdxIqmYUi-cP"
            print("Downloading trained gaussian mixture model")
            download_file_from_google_drive(file_id, self.file_emm)

        self.file_gbc = os.path.join(self.root, self.gbc_path)
        if os.path.exists(self.file_gbc) and not force:
            pass
        else:
            file_id = "15FGLLMZNr55boaGvAdSBMKT2xNEgT3Ij"
            print("Downloading trained gradient boosting classifier")
            download_file_from_google_drive(file_id, self.file_gbc)

    def preprocess_data(self, threshold=0.6, add_negative_sample=True, ratio_neg=1):
        """
        Does the preprocessing:
        - Reading the necessary files.
        - Loading the Gaussian mixture.
        - Creates the dataset based on all the different sources.
        - Loads all this data as class variables.

        Add negative sample creates an artificial negative sample.
        ratio_neg is the proportion used for the initial sampling.
        threshold is the treshold for STITCH score.
        """

        #creates the gold standard from STITCH if not found
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
        dataset = PromisDataset(df_interactions,
                                gaussian_mixture,
                                tanimoto_accessions, tanimoto_global,
                                locations_mets,
                                locations_prots)
        X, y, mets, prots = dataset.get_data()
        self.data = X
        self.labels = y
        self.data_proteins = prots
        self.data_mets = mets

    def _process_external_data(self, df_interactions):
        """
        Gets the data from a external set of interactions to numpy.
        """
        tanimoto_global = pd.read_csv(
            "./data/tanimoto_global.csv", index_col=0)
        tanimoto_accessions = pd.read_csv(
            "./data/tanimoto_accessions.csv", index_col=0)
        with open(self.file_emm, "rb") as f:
            gaussian_mixture = pickle.load(f)
        locations_mets = pd.read_csv(
            "./data/metabolite_locations.csv", sep=",")
        locations_prots = pd.read_csv("./data/proteins_locations.csv", sep=",")
        dataset = PromisDataset(df_interactions,
                                gaussian_mixture,
                                tanimoto_accessions, tanimoto_global,
                                locations_mets,
                                locations_prots)
        X, _, _, _ = dataset.get_data()
        return X

    def _add_negative_sample(self, df, reference_df, ratio=1):
        """
        Adds a negative sample to df.
        reference df is a dataframe used as reference for the positive samples.
        If a pair randomly sampled is contained, it's dropped.
        ratio is the initial number of samples, as a fraction of the length of
        the positive sample (length of df).

        """
        size = int(df.shape[0] * ratio)
        mets = df["chemical"].to_numpy()
        prots = df["protein"].to_numpy()
        met_sample = np.random.choice(mets, size=size)
        prot_sample = np.random.choice(prots, size=size)
        # random sample, with scores init to 0
        random_sample = pd.DataFrame(
            {"chemical": met_sample, "protein": prot_sample, "combined_score": np.zeros([size])})
        # outer join keeping track of the origin.
        random_merged = reference_df.merge(
            random_sample, how='outer', on=["chemical", "protein"], indicator=True)
        # The random pairs found in both dataframes are discarded.
        is_not_repeated = random_merged["_merge"] == "right_only"
        random_sample = random_merged.loc[is_not_repeated]
        random_sample = random_sample.loc[:, [
            "chemical", "protein", "combined_score_y"]]
        random_sample.columns = ["chemical", "protein", "combined_score"]
        # return both dataframes concatenated
        output = pd.concat([df, random_sample], axis=0).reset_index(drop=True)
        output = output.sample(frac=1, replace=False).reset_index(drop=True)
        return output

    def train(self, optimize_hyperparameters=False):
        """
        Trains on all the data.
        If optimize_hyperparameters is set to True. Previously searches for the
        optimal set of hyperparameters.
        """
        if optimize_hyperparameters:
            self.optimize_hyperparameters()
        self._train_on_data()

    def _train_on_data(self, neg_weight=0.9):
        """
        Trains on data and stores the trained model as a class variable.
        """
        y = self.labels
        params = self.hyperparameters()
        num_boost_round = params["num_boost_round"]
        # assigns weights based on the probabilities found in STITCH
        xg_train = xgb.DMatrix(self.data, label=np.round(
            y).astype(np.int64), weight=np.where(y == 0, neg_weight, y))
        model = xgb.train(params, xg_train, num_boost_round=num_boost_round)
        self.gbc = model

    def _cv_gb(self, params, num_boost_round, tgt):
        """
        Cross validates the gradient boosting classifier, and returns the cross-
        validated metric.
        Used for hyperparameter optimization.
        tgt can be loss or auc, the function returns an upper/lower bound.
        """
        features = ["cluster_{}".format(i) for i in range(10)] + \
            ["tanimoto_acc_{}".format(i+1) for i in range(18)] + \
            ["subc_loc_prot_{}".format(i) for i in range(20)] + \
            ["subc_loc_met_{}".format(i+1) for i in range(6)]
        params = params
        params['nthread'] = 12
        num_boost_round = num_boost_round
        y = self.labels
        xg_train = xgb.DMatrix(self.data, label=np.round(
            y).astype(np.int64), weight=np.where(y == 0, 0.9, y))
        model = xgb.cv(params, xg_train,
                       num_boost_round=num_boost_round, nfold=8, shuffle=True)
        # compute an upper bound for the loss.
        lower_ci_acc = (model["test-logloss-mean"] + 3
                        * model["test-logloss-std"]).iloc[model.shape[0]-1]
        # compute a lower bound for the AUC.
        lower_ci_auc = (model["test-auc-mean"] - 3
                        * model["test-auc-std"]).iloc[model.shape[0]-1]
        if tgt == "loss":
            return lower_ci_acc
        elif tgt == "auc":
            return -lower_ci_auc

    def optimize_hyperparameters(self, tgt="loss"):
        """
        Searches the hyperparameter space for the set optimizing the loss or the
        AUC.
        """
        # sets the search space
        space = [Real(10**-8, 10**0, name='gamma'),
                 Real(10**-4, 10**0, name='eta'),
                 Integer(2, 10, name='max_depth'),
                 Real(10**-7, 10**1,  name='reg_lambda'),
                 Real(10**-7, 10**1,  name='alpha'),
                 Real(0.7, 1,  name='subsample'),
                 Integer(10, 500, name='num_boost_round'),
                 Integer(0, 15, name='min_child_weight')]
        # defines the objective

        @use_named_args(space)
        def objective(**params_opt):
            params = {"objective": "binary:logistic",
                      "eval_metric": ["auc", "aucpr", "logloss"],
                      "verbosity": 0}
            for key in params_opt.keys():
                if key not in ['num_boost_rounds', 'early_stopping_rounds']:
                    params[key] = params_opt[key]
            return self._cv_gb(params=params,
                               num_boost_round=params_opt["num_boost_round"], tgt=tgt)
        # calls the minimizer
        res_gp = gp_minimize(objective, space, n_calls=150,
                             n_restarts_optimizer=10, n_jobs=20, kappa=2)
        opt_params = {space[i].name: res_gp["x"][i] for i in range(len(space))}
        # sets the parameters of the model as the optimal
        for key in opt_params.keys():
            self.set(key, opt_params[key])

    def load_trained(self, path=None):
        """
        Loads a pretrained model. If no path is indicated, then downloads the
        model trained during the experiment.
        """
        if path is None:
            with open(self.file_gbc, "rb") as f:
                self = pickle.load(f)
        else:
            with open(path, "rb") as f:
                self = pickle.load(f)

    def save_trained(self, path):
        """
        Saves a trained model to the path specified by the user.
        """
        with open(path, "wb") as f:
            pickle.dump(self, self.gbc)

    def cv_metrics(self, splits=8, neg_weight=0.9):
        """
        Cross-validates a set of metrics about model performance.
        """
        kf = KFold(n_splits=splits, shuffle=True)
        y = self.labels
        # Sets weights to the instances based on the stitch scores and sets the
        # artificial negative set to a weight of neg weight.
        weights = np.where(y == 0, neg_weight, y)
        X = self.data
        y = np.round(y).astype(np.int64)
        params = self.hyperparameters()
        params["eval_metric"] = ["auc", "aucpr", "logloss", "error"]
        params["objective"] = "binary:logistic"
        num_boost_round = params["num_boost_round"]
        # creates the Dmatrix necessary for xgboost.
        xg_train = xgb.DMatrix(
            X, label=y, weight=weights)
        training_df = xgb.cv(params, xg_train,
                             num_boost_round=num_boost_round,
                             folds=kf, shuffle=True)
        return training_df

    def cv_predict(self, splits=8, neg_weight=0.9):
        """
        Predicts all the instances in the internal data by iteratively using cross-validated
        models trained on all the other data.
        """
        kf = KFold(n_splits=splits, shuffle=False)
        y = self.labels
        # weights based on the STITCH scores and neg weight for the negative set.
        weights = np.where(y == 0, neg_weight, y)
        X = self.data
        y = np.round(y).astype(np.int64)
        params = self.hyperparameters()
        params["eval_metric"] = ["auc", "aucpr", "logloss", "error"]
        params["objective"] = "binary:logistic"
        num_boost_round = params["num_boost_round"]
        params.pop("num_boost_round", None)
        preds = []
        for train_index, test_index in kf.split(self.data):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            weights_train, weights_test = weights[train_index], weights[test_index]
            dtrain = xgb.DMatrix(
                X_train, label=y_train, weight=weights_train)
            dtest = xgb.DMatrix(
                X_test, label=y_test, weight=weights_test)
            model = xgb.train(params, dtrain,
                              num_boost_round=num_boost_round)
            preds.append(model.predict(dtest))
        return np.concatenate(preds)

    def _pairwise_predict(self, metabolites: np.array, proteins: np.array) -> pd.DataFrame:
        """
        Returns the pairwise predictions in form of a DataFrame.
        """
        length = metabolites.shape[0]
        scores = np.zeros(length)
        predictions_df = (pd.DataFrame({"chemical": metabolites,
                                       "protein": proteins,
                                        "combined_score": scores})
                          .astype({"chemical": int}))
        X = self._process_external_data(predictions_df)
        dpred = xgb.DMatrix(X)
        preds = self.gbc.predict(dpred)
        predictions_df = predictions_df.assign(combined_score=preds)
        return predictions_df

    def _crosswise_predict(self, metabolites: np.array, proteins: np.array) -> pd.DataFrame:
        """
        Returns the crosswise predictions in form of a DataFrame.
        """
        metabolites = np.unique(metabolites)
        proteins = np.unique(proteins)
        length = metabolites.shape[0] * proteins.shape[0]
        scores = np.zeros(length)
        predictions_df = (pd.DataFrame({"chemical": metabolites})
                          .astype({"chemical": int})
                          .merge(pd.DataFrame({"protein": prots}), how="cross")
                          .assign(combined_score=scores))
        X = self._process_external_data(predictions_df)
        dpred = xgb.DMatrix(X)
        preds = self.gbc.predict(dpred)
        predictions_df = predictions_df.assign(combined_score=preds)
        return predictions_df
