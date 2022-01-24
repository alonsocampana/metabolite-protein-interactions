import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils import *
from datasets import *
from DataHandler import *


class ModelHyperParameters():
    """
    Class for initializing parameters of a model.
    """

    def __init__(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class PROMISHyperParameters(ModelHyperParameters):
    """
    Class for initializing parameters of a GBC model.
    """

    def __init__(self,
                 gamma=1e-07,
                 eta=0.0001,
                 max_depth=66,
                 reg_lambda=2.73,
                 alpha=1e-07,
                 subsample=0.5669498588879256,
                 num_boost_round=383,
                 min_child_weight=2):

        self.parameters = {}
        self.parameter_list = ['gamma', 'eta',
                               'max_depth', 'reg_lambda', 'alpha',
                               'subsample', 'num_boost_round', 'min_child_weight']
        for parameter in self.parameter_list:  # converts the parameters to a dictionary
            self.parameters[parameter] = eval(parameter)

    def __call__(self):
        return self.parameters

    def __str__(self):
        return self.parameters


class End2endModel():
    """
    Base class for models.
    """

    def __init__(self, model_name, root):
        """
        Initializes the name of the model and the root, the parent directory where the needed files will be downloaded.
        """
        self.root = root
        self.model_name = model_name
        self.loaded = False
        self.trained = False
        self._check_directory(True)
        self._download_and_preprocess()
        self.tanimoto_global_path = "./data/tanimoto_global.csv"
        self.tanimoto_accessions_path = "./data/tanimoto_accessions.csv"
        self._preprocess_tanimotos()
        self.goldstandard_path = os.path.join(
            self.root, "goldstandard_PROMIS.csv")

    def set(self, parameter, value):
        """
        Method for setting the hyperparameters of the model.
        If the hyperparameter is not valid will return an error.
        """
        if parameter not in self.hyperparameters.parameter_list:
            print("{} is not a parameter".format(parameter))
            raise KeyError
        else:
            self.hyperparameters.parameters[parameter] = value

    def _download_and_preprocess(self):
        """
        Downloads and preprocesses the general data needed for the models.
        """
        data = os.path.join(os.path.dirname(self.root), "data")
        # Create instances of handlers to automatically download and preprocess the data
        AAHandler(data)
        UniprotHandler(data)
        PROMISProteinsHandler(data)
        PROMISMetabolitesHandler(data)
        STITCHandler(data)
        ChemicalPropertiesHandler(data)
        ProtLocsHandler(data)
        MetLocsHandler(data)

    def _check_directory(self, force=True):
        """
        Checks if the root exists.
        If it doesn't exist it creates the folder.
        """
        if not os.path.exists(self.root):
            if force:
                os.makedirs(self.root)
            else:
                raise FileNotFoundError

    def _preprocess_tanimotos(self, force=False):
        """
        Reads the PROMIS profile data, and creates a new csv file where the tanimoto indices of the normalized profiles is contained.
        """
        if os.path.exists(self.tanimoto_global_path) \
                and os.path.exists(self.tanimoto_accessions_path) \
                or force:
            pass
        else:
            prots_df = pd.read_csv(
                "./data/protein_profiles_processed.csv", index_col=0)
            mets_df = pd.read_csv(
                "./data/metabolite_profiles_processed.csv", index_col=0)
            tanimoto_df = self._get_tanimoto_df(mets_df, prots_df)
            shp = tanimoto_df.shape
            # Reshapes in function of the number of accessions (18) and
            # fractions (10) per accession to vectorize the operation
            tanimoto_accessions = (tanimoto_df
                                   .iloc[:, 0:180]
                                   .to_numpy()
                                   .reshape([shp[0] * 18,
                                            int(shp[1]/18)])
                                   .sum(axis=1)
                                   .reshape(shp[0],
                                            int(shp[1]/10)))
            # gets rid of possible duplicates
            tanimoto_df.groupby(["metabolite", "protein"]).min(
            ).reset_index().to_csv(self.tanimoto_global_path)
            # extract accession name via regex
            accessions = tanimoto_df.columns.str.extract(
                "([a-zA-Z0-9]+)_").value_counts().index
            tanimoto_accessions_df = pd.DataFrame(tanimoto_accessions, columns=[
                                                  acc[0] for acc in accessions])
            tanimoto_accessions_df = tanimoto_accessions_df.assign(metabolite=tanimoto_df.loc[:, "metabolite"].to_numpy(),
                                                                   protein=tanimoto_df.loc[:, "protein"].to_numpy())
            # gets rid of possible duplicates
            tanimoto_accessions_df.groupby(["metabolite", "protein"]).min(
            ).reset_index().to_csv(self.tanimoto_accessions_path)

    def _get_tanimoto_df(self, mets_df: pd.DataFrame, prots_df: pd.DataFrame):
        """
        Given a dataframe containing the metabolite profiles and a dataframe
         containing the proteins profiles, returns the tanimoto indices of the
         different accessions.
        """
        tani_df = pd.DataFrame(self._tanimoto_profiles(
            mets_df.iloc[:, 2:].to_numpy(), prots_df.iloc[:, 1:].to_numpy()))
        tani_df.columns = mets_df.iloc[:, 2:].columns
        tani_df = (tani_df.assign(metabolite=np.tile(mets_df["cid"].to_numpy().reshape([mets_df.shape[0], 1]),
                                                     (1, prots_df.shape[0], ))
                                  .reshape([mets_df.shape[0] * prots_df.shape[0]]))
                   .assign(protein=np.tile(prots_df["String"].to_numpy(),
                                           (mets_df.shape[0], 1))
                           .reshape([mets_df.shape[0] * prots_df.shape[0]])))
        return tani_df

    def _generate_gold_standard(self, threshold):
        """
        Generates the gold standard used for the PROMIS data, taking the STITCH
        data and selecting the pairs found in the PROMIS experiment above a
        defined threshold. Along the experiment 0.6 was used as threshold.
        """
        df_interactions = pd.read_csv("./data/STITCH_processed.csv")
        tanimoto_global = pd.read_csv(
            "./data/tanimoto_global.csv", index_col=0)
        proteins = tanimoto_global["protein"].unique()
        metabolites = tanimoto_global["metabolite"].unique()
        gt_threshold = df_interactions["combined_score"] > threshold
        is_protein_in_data = df_interactions["protein"].isin(proteins)
        is_chemical_in_data = df_interactions["chemical"].isin(metabolites)
        df_interactions = df_interactions[is_protein_in_data
                                          & is_chemical_in_data & gt_threshold]
        df_interactions.to_csv(self.goldstandard_path)
        return df_interactions

    def _tanimoto_profiles(self, mets: np.ndarray, prots: np.ndarray):
        """
        Returns the tanimoto index, by returning the minimum value for each ij
         entry. Where i is the accession and j the fraction.
        """
        mets_shp = mets.shape
        prots_shp = prots.shape
        mets_repeated = np.repeat(np.expand_dims(mets, 1), prots_shp[0], axis=1).reshape(
            [mets_shp[0] * prots_shp[0], mets_shp[1]])
        prots_repeated = np.tile(prots, (mets_shp[0], 1))
        return(np.minimum(mets_repeated, prots_repeated))

    def preprocess_data(self):
        """
        Preprocessing pipeline for a given model.
        """
        raise NotImplementedError

    def train(self, optimize_hyperparameters=False):
        """
        Training loop for a given model. optimize_hyperparameters specifies if
        the training should be done with the initial hyperparameters or the optimal
        set should be found.
        """
        raise NotImplementedError

    def predict_pairs(self, metabolites: np.array,
                      proteins: np.array, how="pair"):
        """
        Predicts the pairs found in a set of metabolites and proteins.
        If how == pair, then they should be same length, because they'll be
        treated as pairs. If how cross it will perform the cross product of both
        sets and predict all possible interactions.
        """
        assert how in ["pair", "cross"]
        if how == "pair":
            assert metabolites.shape == proteins.shape
            preds = self._pairwise_predict(metabolites, proteins)
        if how == "cross":
            preds = self._crosswise_predict(metabolites, proteins)
        return preds

    def cv_metrics(self):
        """
        Cross-validated metrics about the model.
        """
        raise NotImplementedError

    def cv_predict(self):
        """
        Return the predictions of all the data for the holdout data.
        """
        raise NotImplementedError

    def load_trained(self):
        """
        Loads a pretrained model
        """
        raise NotImplementedError

    def plot_roc(self, y_pred, y_true, ax, color="red"):
        """
        Plots a ROC curve using matplotlib, and a given set of true labels and
        predictions.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        p = plt.plot(
            fpr,
            tpr,
            color=color,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        return p

    def __str__(self):
        print(self.model_name, "\t")
        print(self.hyperparameters())
