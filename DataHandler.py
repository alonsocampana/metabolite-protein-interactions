## %
import os
import requests
import urllib
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.preprocessing import MinMaxScaler
import base64
from utils import *


def normalize_profiles(profile: np.ndarray, profile_length=10, n_accessions=18, eps=0.00001) -> np.array:
    """
    From a given set of chromatographic profiles, returns the accession-wise normalized profiles.
    [profile_length]: Number of fractions found in each profile
    [n_accessions]: Number of accessions found in the data
    """
    shp = profile.shape
    profile_reshaped = profile.reshape(
        [shp[0] * n_accessions, int(shp[1]/n_accessions)])
    profile_normalized = profile_reshaped / \
        (np.repeat(np.expand_dims(profile_reshaped.sum(
            axis=1) + eps, 1), profile_length, axis=1))
    profile_normalized = profile_normalized.reshape(shp)
    return profile_normalized


class DataHandler():
    """
    Interface for the DataHandlers.
    """

    def __init__(self, root="./data/", filename="<name>", filename_processed="<name>"):

        self.root = root
        self._check_directory()
        self.filename = filename
        self.filename_processed = filename_processed

    def get(self, force=False):
        """
        Downloads the needed files to the root.
        """
        self.full_path = os.path.join(self.root, self.filename)
        if os.path.exists(self.full_path) and not force:
            return "File already exists!"
        else:
            self._get_pipeline()

    def preprocess(self, force=False):
        """
        Performs the shared preprocessing of the files and stores them in the root.
        """
        self.full_path_processed = os.path.join(
            self.root, self.filename_processed)
        if os.path.exists(self.full_path_processed) and not force:
            return "File already exists!"
        else:
            self._preprocessing_pipeline()

    def _get_pipeline(self):
        raise NotImplementedError

    def _preprocessing_pipeline(self):
        raise NotImplementedError

    def _check_directory(self, force=True):
        if not os.path.exists(self.root):
            if force:
                os.makedirs(self.root)
            else:
                raise FileNotFoundError


class STITCHandler(DataHandler):
    def __init__(self, root="./data/"):
        """
        DataHandler for downloading the STITCH database and preprocessing it.
        [root]: The path for the data. If it doesn't exist will be created.
        """
        filename = 'STITCH_raw.tsv.gz'
        filename_processed = 'STITCH_processed.csv'
        super().__init__(root, filename, filename_processed)
        self.url = "http://stitch.embl.de/download/protein_chemical.links.detailed.v5.0/3702.protein_chemical.links.detailed.v5.0.tsv.gz"
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading STITCH data...")
        urllib.request.urlretrieve(
            self.url, self.full_path)  # Downloads the data

    def _preprocessing_pipeline(self):
        stitch = pd.read_csv(self.full_path, sep="\t")
        stitch = stitch.loc[:, ["protein", "chemical", "combined_score"]]
        stitch = stitch[stitch["combined_score"] > 500]
        stitch.loc[:, "combined_score"] = stitch["combined_score"]/1000
        stitch = stitch.assign(chemical=stitch["chemical"].str.extract(
            "CID[ms][0]*([1-9][0-9]*)").astype(int))  # Stitch CID to pubchem CID
        # gets rid of possible duplicates
        stitch = stitch.groupby(["chemical", "protein"]).min()
        stitch = stitch.reset_index()
        stitch.to_csv(self.full_path_processed, index=False)  # saves the file


class AAHandler(DataHandler):
    """
    Interface for downloading AA properties from AAindex
    Example usage:
    AA = AAHandler()
    """

    def __init__(self, root="./data/", scale=10):
        self.scale = scale
        filename = 'AAindex_raw.csv'
        filename_processed = 'AAindex_processed.csv'
        super().__init__(root, filename, filename_processed)
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading AAindex data...")
        download_file_from_google_drive(
            "14oY9g3VO1hvrnmyJ2qvtMvJvLYX53nWR", self.full_path)

    def _preprocessing_pipeline(self, n_props=128):
        AA_df = pd.read_csv(self.full_path, index_col=0)
        mms = MinMaxScaler([-self.scale, self.scale])
        # transforms the data using min max scaling
        AA_df.iloc[:, :] = mms.fit_transform(AA_df)
        AA_df.to_csv(self.full_path_processed)
        props = AA_df.to_numpy()[:, 0:n_props]
        AA_dict = {AA_df.index.to_numpy()[i]: props[i]
                   for i in range(len(props))}
        AA_dict["X"] = np.zeros(n_props)
        with open("./data/AAdict.pkl", "wb") as f:
            pickle.dump(AA_dict, f)


class UniprotHandler(DataHandler):
    """
    Interface for Uniprot data
    """

    def __init__(self, root="./data/"):
        filename = 'uniprot_raw.tab.gz'
        filename_processed = 'uniprot_processed.csv'
        super().__init__(root, filename, filename_processed)
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading Uniprot protein data...")
        download_file_from_google_drive(
            "1A0HTCWygdrwRyNNlrCE8sn8ZCOyYfPVk", self.full_path)

    def _preprocessing_pipeline(self):
        uniprot = pd.read_csv("./data/uniprot_raw.tab.gz",
                              compression="gzip", sep="\t")
        # Clean up the STRING column
        uniprot["Cross-reference (STRING)"] = uniprot["Cross-reference (STRING)"].str.replace(";", "")
        uniprot.to_csv(self.full_path_processed)


class PROMISMetabolitesHandler(DataHandler):
    """
    Interface for PROMIS profiles of the metabolites.
    """

    def __init__(self, root="./data/"):
        self.root = root
        filename = 'metabolite_profiles_raw.csv'
        filename_processed = 'metabolite_profiles_processed.csv'
        super().__init__(root, filename, filename_processed)
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading PROMIS metabolite profiles...")
        download_file_from_google_drive(
            "1essmFw23B3ek6ofeR3EW7OswYB8sTHcf", self.full_path)

    def _preprocessing_pipeline(self):
        metabolites = pd.read_csv(self.full_path, index_col=0)
        # Replace columns by normalized profiles
        metabolites.iloc[:, 2:] = normalize_profiles(
            metabolites.iloc[:, 2:].to_numpy())
        metabolites.to_csv(self.full_path_processed)


class PROMISProteinsHandler(DataHandler):
    """
    Interface for PROMIS profiles of the proteins.
    """

    def __init__(self, root="./data/"):
        self.root = root
        filename = 'protein_profiles_raw.csv'
        filename_processed = 'protein_profiles_processed.csv'
        super().__init__(root, filename, filename_processed)
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading PROMIS protein profiles...")
        download_file_from_google_drive(
            "12lN1FL6TjyTHKo1rK98QgaFB6S_Vseho", self.full_path)

    def _preprocessing_pipeline(self):
        proteins = pd.read_csv(self.full_path, index_col=0)
        proteins.iloc[:, 1:] = normalize_profiles(
            proteins.iloc[:, 1:].to_numpy())
        proteins.to_csv(self.full_path_processed)


class ChemicalPropertiesHandler(DataHandler):
    """
    Interface for metabolite features from pubchem.
    """

    def __init__(self, root="./data/"):
        self.root = root
        filename = 'chemicals_properties_raw.csv'
        filename_processed = 'chemicals_properties_processed.csv'
        super().__init__(root, filename, filename_processed)
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading chemical properties...")
        download_file_from_google_drive(
            "15yRiPNLGIWR3oFeR6c-4K8ei25j3wN2v", self.full_path)

    def _preprocessing_pipeline(self):
        metabolite_properties = pd.read_csv(self.full_path, index_col=0)
        # decodes the Fingerprint into binary features
        fingerprint_features = metabolite_properties["Fingerprint2D"].apply(
            self._to_bits_decode)
        length = metabolite_properties.shape[0]
        threshold = length//500
        # Get rid of features containing (almost) no information
        is_fp_always_zero = fingerprint_features.sum() < threshold
        is_fp_always_one = fingerprint_features.sum() > length - threshold
        is_informative = ~is_fp_always_zero & ~is_fp_always_one
        fingerprint_features = pd.DataFrame(
            np.stack(fingerprint_features.values))
        metabolite_properties = pd.concat([metabolite_properties
                                           .drop("Fingerprint2D", axis=1).reset_index(
                                               drop=True), fingerprint_features.iloc[:, is_informative]], axis=1)
        # If feature is na, replace by mean and create dummy feature
        cols_to_dummy = metabolite_properties.loc[:, metabolite_properties.isna(
        ).sum() > 0].columns.to_numpy()
        for col in cols_to_dummy:
            colmean = metabolite_properties[col].mean()
            is_col_na = metabolite_properties[col].isna().astype(int)
            metabolite_properties[col+"is_na"] = is_col_na
            metabolite_properties[col] = metabolite_properties[col].replace(
                np.nan, colmean)
        metabolite_properties.to_csv(self.full_path_processed)

    def _to_bits_decode(self, seq):
        """
        Helper function for decoding bit data.
        """
        length = len(seq)
        decoded = base64.decodebytes(seq.encode('utf-8'))
        bits = []
        for x in decoded:
            bits += [i for i in str("{:08b}".format(x))]
        return(np.array(bits).astype(int))


class ProtLocsHandler(DataHandler):
    """
    Interface for downloading protein subcellular location data.
    """

    def __init__(self, root="./data/"):
        self.root = root
        filename = 'proteins_locations.csv'
        filename_processed = ''
        super().__init__(root, filename, filename_processed)
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading protein subcellular locations...")
        download_file_from_google_drive(
            "1GQEEScySC3snyFqwe87Wyy4fvRylXMOY", self.full_path)

    def _preprocessing_pipeline(self):
        pass


class MetLocsHandler(DataHandler):
    """
    Interface for downloading metabolite subcellular location data.
    """

    def __init__(self, root="./data/"):
        self.root = root
        filename = 'metabolite_locations.csv'
        filename_processed = ''
        super().__init__(root, filename, filename_processed)
        self.get()
        self.preprocess()

    def _get_pipeline(self):
        print("Downloading metabolite subcellular locations...")
        download_file_from_google_drive(
            "1p25fZx1qLgeCQ4iS7bI5XHmpbnv6b6Oi", self.full_path)

    def _preprocessing_pipeline(self):
        pass
