import numpy as np
import torch
from utils import *
import pickle
import pandas as pd


class PromisDataset(torch.utils.data.Dataset):
    def __init__(self, df_interactions, gaussian_mixture, tanimoto_profiles, tanimoto_df, locations_mets, locations_prots, transform=None, target_transform=None):
        self.df_interactions = df_interactions.copy(
            deep=True).dropna().drop_duplicates().groupby(["chemical", "protein"]).min().reset_index().sample(frac=1)
        self.gaussian_mixture = gaussian_mixture
        self.tanimoto_profiles = tanimoto_profiles.drop_duplicates()
        self.tanimoto_df = tanimoto_df.drop_duplicates()
        self.locations_mets = locations_mets.drop_duplicates()
        self.locations_prots = locations_prots.drop_duplicates()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.df_interactions.shape[0]

    def get_data(self):
        mets = self.df_interactions["chemical"].astype(int)
        prots = self.df_interactions["protein"]
        tgts = self.df_interactions["combined_score"]
        emm_preds = self.gaussian_mixture.predict(self.df_interactions
                                                  .astype({"chemical": int})
                                                  .merge(self.tanimoto_df,
                                                         right_on=[
                                                             "metabolite", "protein"],
                                                         left_on=["chemical", "protein"], how="left")
                                                  .replace(np.nan, 0).iloc[:, 4:184].to_numpy())
        clusters = np.zeros([emm_preds.shape[0], 10])
        clusters[np.arange(0, emm_preds.shape[0]), emm_preds] = 1
        tanimotos = (self.df_interactions
                     .astype({"chemical": int})
                     .merge(self.tanimoto_profiles,
                            right_on=["metabolite", "protein"],
                            left_on=["chemical", "protein"], how="left")
                     .iloc[:, 4:].to_numpy())
        geometric_mean_global = np.prod(tanimotos, axis=1)[:, None]
        max_tanimoto = np.max(tanimotos, axis=1)[:, None]
        min_tanimoto = np.min(tanimotos, axis=1)[:, None]
        geo_mean_max3 = np.prod(np.sort(tanimotos, axis=1)[
                                :, ::-1][:, :3], axis=1)[:, None]
        geo_mean_max9 = np.prod(np.sort(tanimotos, axis=1)[
                                :, ::-1][:, :9], axis=1)[:, None]
        geo_mean_min3 = np.prod(np.sort(tanimotos, axis=1)[
                                :, :3], axis=1)[:, None]
        geo_mean_min9 = np.prod(np.sort(tanimotos, axis=1)[
                                :, :9], axis=1)[:, None]
        tanimotos = np.concatenate([tanimotos, geometric_mean_global, max_tanimoto,
                                   min_tanimoto, geo_mean_max3, geo_mean_max9, geo_mean_min3, geo_mean_min9], axis=1)

        loc_mets = (self.df_interactions
                    .merge(self.locations_mets,
                           left_on="chemical",
                           right_on="CID", how="left")
                    .replace(np.nan, 0)[["Plastid", "Mitochondria", "Peroxisome", "ER", "Vacuole", "Golgi"]])
        loc_prots = (self.df_interactions
                     .merge(self.locations_prots, left_on="protein", right_on=[
                                               "Cross-reference (STRING)"], how="left").replace(np.nan, 0).iloc[:, 4:].to_numpy())
        self.data = np.concatenate(
            [clusters, tanimotos, loc_mets, loc_prots[:, 1:]], axis=1)
        self.tgt = tgts
        return self.data, tgts.to_numpy(), mets, prots


class ProtPairsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df_interactions,
                 uniprot,
                 met_features,
                 weights,
                 transform=None,
                 target_transform=None):
        self.uniprot = uniprot
        self.df_interactions = df_interactions
        self.met_features = met_features
        self.weights = weights
        self.transform = transform
        self.target_transform = target_transform
        self.positional_encoding = self.get_positional_encoding(512, 128)
        with open("./data/AAdict.pkl", "rb") as f:
            self.AA_dict = pickle.load(f)

    def __len__(self):
        return self.df_interactions.shape[0]

    def __getitem__(self, idx):
        entry = self.df_interactions.iloc[idx]
        met = entry["chemical"]
        prot = entry["protein"]
        tgt = (entry["combined_score"])
        tgt = np.array(tgt).astype(np.float32)
        seq = self.uniprot.loc[prot, "Sequence"]
        if tgt > 0:
            weight = np.array(self.weights.loc[prot, "weight"])
        else:
            weight = np.array(1).astype(np.float32)
        encoding, _ = self.embed_sequence(seq)
        met_features = self.met_features.loc[met].to_numpy()
        return encoding, met_features, tgt, weight

    def embed_sequence(self, seq, max_length=512):
        seq = list(seq)
        posrr = np.array([self.AA_dict[aa][0:128] for aa in seq]).T
        length = posrr.shape[1]
        if length > max_length-2:
            posrr = posrr[:, :max_length-2]
            mask = np.concatenate(
                [np.zeros(1), np.ones(max_length-2), np.zeros(1)])
        else:
            mask = np.concatenate([np.zeros(1), np.ones(
                length), np.zeros(max_length-1-length)])
        posrr = np.concatenate(
            [np.zeros([128, 1]), posrr, np.zeros([128, 1])], axis=1)
        prot_arr = np.zeros([128, max_length])
        prot_arr[:posrr.shape[0], :posrr.shape[1]] = posrr
        final_embedding = (prot_arr + self.positional_encoding).T
        return final_embedding, mask

    def get_positional_encoding(self, seq_length, n_features):
        positional_encoding_1 = np.sin(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding_2 = np.cos(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding = np.concatenate(
            [positional_encoding_1, positional_encoding_2], axis=0)
        return positional_encoding


class ProtPairsDatasetPred(torch.utils.data.Dataset):
    def __init__(self,
                 df_interactions,
                 uniprot,
                 met_features,
                 transform=None,
                 target_transform=None):
        self.uniprot = uniprot
        self.df_interactions = df_interactions
        self.met_features = met_features
        self.transform = transform
        self.target_transform = target_transform
        self.positional_encoding = self.get_positional_encoding(512, 128)
        with open("./data/AAdict.pkl", "rb") as f:
            self.AA_dict = pickle.load(f)

    def __len__(self):
        return self.df_interactions.shape[0]

    def __getitem__(self, idx):
        entry = self.df_interactions.iloc[idx]
        met = entry["chemical"]
        prot = entry["protein"]
        seq = self.uniprot.loc[prot, "Sequence"]
        encoding, _ = self.embed_sequence(seq)
        met_features = self.met_features.loc[met].to_numpy()
        return encoding, met_features

    def embed_sequence(self, seq, max_length=512):
        seq = list(seq)
        posrr = np.array([self.AA_dict[aa][0:128] for aa in seq]).T
        length = posrr.shape[1]
        if length > max_length-2:
            posrr = posrr[:, :max_length-2]
            mask = np.concatenate(
                [np.zeros(1), np.ones(max_length-2), np.zeros(1)])
        else:
            mask = np.concatenate([np.zeros(1), np.ones(
                length), np.zeros(max_length-1-length)])
        posrr = np.concatenate(
            [np.zeros([128, 1]), posrr, np.zeros([128, 1])], axis=1)
        prot_arr = np.zeros([128, max_length])
        prot_arr[:posrr.shape[0], :posrr.shape[1]] = posrr
        final_embedding = (prot_arr + self.positional_encoding).T
        return final_embedding, mask

    def get_positional_encoding(self, seq_length, n_features):
        positional_encoding_1 = np.sin(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding_2 = np.cos(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding = np.concatenate(
            [positional_encoding_1, positional_encoding_2], axis=0)
        return positional_encoding


class PromisDatasetXGB(torch.utils.data.Dataset):
    def __init__(self, df_interactions,
                 gaussian_mixture,
                 tanimoto_profiles,
                 tanimoto_df,
                 locations_mets,
                 locations_prots,
                 df_predictions,
                 transform=None,
                 target_transform=None):
        self.df_interactions = df_interactions.copy(deep=True)
        self.gaussian_mixture = gaussian_mixture
        self.tanimoto_profiles = tanimoto_profiles
        self.tanimoto_df = tanimoto_df
        self.locations_mets = locations_mets
        self.locations_prots = locations_prots
        self.df_predictions = df_predictions
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.df_interactions.shape[0]

    def get_data(self):
        mets = self.df_interactions["chemical"]
        prots = self.df_interactions["protein"]
        multiidx = pd.MultiIndex.from_arrays([mets, prots])
        tgts = self.df_interactions["combined_score"]
        preds = self.df_predictions.loc[multiidx]["combined_score"].to_numpy()
        emm_preds = self.gaussian_mixture.predict(self.df_interactions
                                                  .astype({"chemical": int})
                                                  .merge(self.tanimoto_df,
                                                         right_on=[
                                                             "metabolite", "protein"],
                                                         left_on=["chemical", "protein"], how="left")
                                                  .replace(np.nan, 0).iloc[:, 4:184].to_numpy())
        clusters = np.zeros([emm_preds.shape[0], 10])
        clusters[np.arange(0, emm_preds.shape[0]), emm_preds] = 1
        tanimotos = (self.df_interactions
                     .astype({"chemical": int})
                     .merge(self.tanimoto_profiles,
                            right_on=["metabolite", "protein"],
                            left_on=["chemical", "protein"], how="left")
                     .iloc[:, 4:].to_numpy())
        geometric_mean_global = np.prod(tanimotos, axis=1)[:, None]
        max_tanimoto = np.max(tanimotos, axis=1)[:, None]
        min_tanimoto = np.min(tanimotos, axis=1)[:, None]
        geo_mean_max3 = np.prod(np.sort(tanimotos, axis=1)[
                                :, ::-1][:, :3], axis=1)[:, None]
        geo_mean_max9 = np.prod(np.sort(tanimotos, axis=1)[
                                :, ::-1][:, :9], axis=1)[:, None]
        geo_mean_min3 = np.prod(np.sort(tanimotos, axis=1)[
                                :, :3], axis=1)[:, None]
        geo_mean_min9 = np.prod(np.sort(tanimotos, axis=1)[
                                :, :9], axis=1)[:, None]
        tanimotos = np.concatenate([tanimotos, geometric_mean_global, max_tanimoto,
                                   min_tanimoto, geo_mean_max3, geo_mean_max9, geo_mean_min3, geo_mean_min9], axis=1)
        loc_mets = (self.df_interactions
                    .merge(self.locations_mets,
                           left_on="chemical",
                           right_on="CID", how="left")
                    .replace(np.nan, 0)[["Plastid", "Mitochondria", "Peroxisome", "ER", "Vacuole", "Golgi"]])
        loc_prots = self.df_interactions.merge(self.locations_prots, left_on="protein", right_on=[
                                               "Cross-reference (STRING)"], how="left").replace(np.nan, 0).iloc[:, 4:].to_numpy()
        self.data = np.concatenate(
            [clusters, tanimotos, loc_mets, loc_prots[:, 1:], preds[:, None]], axis=1)
        self.tgt = tgts
        return self.data, tgts, mets.to_numpy(), prots.to_numpy()
