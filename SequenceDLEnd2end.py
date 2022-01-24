import pubchempy
import pandas as pd
import numpy as np
import os
from torch import nn
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from utils import *
from datasets import *
from End2EndModel import *


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


class TransformerEncoderGated(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, **kwargs):
        super().__init__(encoder_layer, num_layers, **kwargs)
        self.gates = nn.Parameter(torch.Tensor(num_layers))
        self.layers = nn.ModuleList([encoder_layer for i in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        range_gates = torch.sigmoid(self.gates)
        for i, layer in enumerate(self.layers):
            src = (range_gates[i])*layer(src) + (1-range_gates[i])*src
        return src


class TransformerProt(nn.Module):
    def __init__(self):
        super().__init__()
        tl = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dropout=0.2, batch_first=True)
        self.transformer = TransformerEncoderGated(tl, 4)
        self.fc = nn.Sequential(nn.Linear(128, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 128),
                                nn.ReLU(),
                                nn.Linear(128, 23))
        self.transformer.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = F.dropout2d(x,  p=0.1)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_size, hidden_space, dropout=0.1):
        super().__init__()
        self.nw = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(input_size, hidden_space),
                                nn.ReLU(),
                                nn.Linear(hidden_space, input_size))

    def forward(self, x):
        x = self.nw(x)
        return x


class N_ResBlocks(nn.Module):
    def __init__(self, first_hidden_dim, hidden_depth, n_blocks, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResBlock(first_hidden_dim, hidden_depth, dropout) for i in range(n_blocks)])
        self.gates = nn.Parameter(torch.Tensor(n_blocks))

    def forward(self, x):
        range_gates = torch.sigmoid(self.gates)
        for i, layer in enumerate(self.layers):
            x = (range_gates[i])*layer(x) + (1-range_gates[i])*x
        return x


class MetEncoder(nn.Module):
    def __init__(self, init_dim, target_dim, p_dropout):
        super().__init__()
        self.nw = nn.Sequential(nn.Linear(init_dim, target_dim*4),
                                nn.ReLU(),
                                nn.BatchNorm1d(target_dim*4),
                                nn.Linear(target_dim*4, target_dim*2),
                                nn.ReLU(),
                                nn.Dropout(p=p_dropout),
                                nn.Linear(target_dim*2, target_dim))

    def forward(self, x):
        return self.nw(x)


class AttnBlock(nn.Module):
    def __init__(self, n_features, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.fc_prot = nn.Sequential(nn.Linear(n_features, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, n_features))
        self.fc_met = nn.Sequential(nn.Linear(n_features, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, n_features))

    def forward(self, x, y):
        x = self.fc_met(x)
        y = self.fc_prot(y)
        z = F.softmax(torch.bmm(y, x.unsqueeze(2))/np.sqrt(x.shape[1]), dim=1)
        return torch.bmm(z.transpose(2, 1), y).squeeze()


class N_AttnBlock(nn.Module):
    def __init__(self, n_features, seq_length, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttnBlock(n_features, seq_length) for i in range(n_blocks)])

    def forward(self, x, y):
        y_outs = [torch.mean(y, axis=1)]
        for block in self.blocks:
            temp_y = block(x, y)
            if len(temp_y.shape) == 1:
                temp_y = temp_y[None, :]
            y_outs.append(temp_y)
        return torch.concat(y_outs, axis=1)


class IntegrativeModel(nn.Module):
    def __init__(self,
                 seq_length,
                 init_dim_met,
                 dim_prot,
                 n_attn_blocks,
                 n_res_blocks,
                 p_dropout,
                 warm=False):
        super().__init__()
        tl = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dropout=p_dropout, batch_first=True)
        self.prot_transformer = TransformerEncoderGated(tl, 4)
        self.met_encoder = MetEncoder(init_dim_met, dim_prot, p_dropout)
        self.attn = N_AttnBlock(dim_prot, seq_length, n_attn_blocks)
        dim_res = dim_prot * (n_attn_blocks + 2)
        self.integrative = nn.Sequential(
            N_ResBlocks(dim_res, dim_res * 2, n_res_blocks))
        self.classifier = nn.Sequential(nn.Dropout(p=p_dropout),
                                        nn.Linear(dim_res, 1))
        self.met_encoder.apply(init_weights)
        self.prot_transformer.apply(init_weights)
        self.attn.apply(init_weights)
        self.integrative.apply(init_weights)
        self.classifier.apply(init_weights)

    def set_cold(self):
        for p in self.prot_transformer.parameters():
            p.requires_grad = False

    def set_warm(self):
        for p in self.prot_transformer.parameters():
            p.requires_grad = True

    def forward(self, x, y):
        x = self.met_encoder(x)
        y = self.prot_transformer(y)
        y = self.attn(x, y)
        x = self.integrative(torch.concat([x, y], axis=1))
        x = self.classifier(x)
        return x


class SequenceDLEnd2end(End2endModel):
    def __init__(self, model_name="PROMISGBC", root="./saved_models"):
        super().__init__(model_name, root)
        self.model_file = "torch_model.pkl"
        self._download_files()
        self._init_model()

    def _download_files(self, force=False):
        self.file_path = os.path.join(self.root, self.model_file)
        if os.path.exists(self.file_path) and not force:
            pass
        else:
            file_id = "170hJbUwvnxNzRJ41W7QTICQ-c3oSvbjh"
            print("Downloading trained torch model")
            download_file_from_google_drive(file_id, self.file_path)

    def _init_model(self,
                    lr=0.00015,
                    wd=0.00000625,
                    n_attn_blocks=16,
                    grad_clip=2,
                    p_dropout=0.15,
                    seed=0,
                    force_cpu=False):

        self.loss_fn = nn.BCEWithLogitsLoss
        self.lr = lr
        self.wd = wd
        self.n_attn_blocks = n_attn_blocks
        self.grad_clip = grad_clip
        self.p_dropout = p_dropout
        torch.manual_seed(seed)

        self.model = IntegrativeModel(seq_length=512,
                                      init_dim_met=628,
                                      dim_prot=128,
                                      n_attn_blocks=n_attn_blocks,
                                      n_res_blocks=2,
                                      p_dropout=p_dropout,
                                      warm=False)

        params_to_optimize = [
            {'params': self.model.parameters()}
        ]
        self.optim = torch.optim.Adam(
            params_to_optimize, lr=lr, weight_decay=wd)
        # Check if the GPU is available
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() and not force_cpu \
            else torch.device("cpu")
        print(f'Selected device: {self.device}')

        self.model.to(self.device)

    def preprocess_data(self, threshold=0.6, add_negative_sample=True, ratio_neg=1, force=False):
        df_interactions = pd.read_csv("./data/STITCH_processed.csv")
        gt_threshold = df_interactions["combined_score"] > threshold
        self.df_interactions = df_interactions[gt_threshold]
        if os.path.exists(self.goldstandard_path) and not force:
            interactions_promis = pd.read_csv(self.goldstandard_path)
        else:
            interactions_promis = self._generate_gold_standard(threshold)
        merged_interactions = df_interactions.merge(
            interactions_promis, on=["chemical", "protein"], how="left", indicator=True)
        self._train_split = df_interactions[merged_interactions["_merge"]
                                            == "left_only"]
        self._test_split = interactions_promis

    def get_dataloader(self, df_interactions, reference_df, batch_size=4, add_negative_sample=True):
        uniprot = pd.read_csv("./data/uniprot_processed.csv")
        uniprot["Cross-reference (STRING)"] = uniprot["Cross-reference (STRING)"].str.replace(";", "")
        uniprot = uniprot.set_index("Cross-reference (STRING)", drop=False)
        chemical_features = pd.read_csv("./data/chemicals_properties_processed.csv",
                                        index_col=0).set_index("CID")
        mms = MinMaxScaler([-1, 1])
        chemical_features.iloc[:, :23] = mms.fit_transform(
            chemical_features.iloc[:, :23].to_numpy())
        weights = self._get_protein_weights(self.df_interactions)
        interaction_set = self._add_negative_sample(
            df_interactions, reference_df)
        dataset = ProtPairsDataset(interaction_set,
                                   uniprot,
                                   chemical_features,
                                   weights)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=20, shuffle=add_negative_sample, drop_last=True)
        self.last_interactions = interaction_set
        return loader

    def get_dataloader_pred(self, df_interactions, batch_size=4):
        uniprot = pd.read_csv("./data/uniprot_processed.csv")
        uniprot["Cross-reference (STRING)"] = uniprot["Cross-reference (STRING)"].str.replace(";", "")
        uniprot = uniprot.set_index("Cross-reference (STRING)", drop=False)
        chemical_features = pd.read_csv("./data/chemicals_properties_processed.csv",
                                        index_col=0).set_index("CID")
        mms = MinMaxScaler([-1, 1])
        chemical_features.iloc[:, :23] = mms.fit_transform(
            chemical_features.iloc[:, :23].to_numpy())
        dataset = ProtPairsDatasetPred(df_interactions,
                                       uniprot,
                                       chemical_features)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=20, shuffle=False, drop_last=False)
        return loader

    def _torch_train(self, dataloader, batch_acc=64, weight_neg=0.9, grap_clip=2):
        self.optimizer.zero_grad()
        self.model.train()
        train_losses = []
        rocs = []
        for x, batch in enumerate(dataloader):
            data_prot = batch[0].float().to(self.device)
            data_met = batch[1].float().to(self.device)
            weights = batch[2].float()
            target = weights.round()
            weights[weights == 0] = weight_neg
            weights = weights.mul(batch[3].squeeze())
            weights = weights.to(self.device)
            target = target.to(self.device)
            loss = self.loss_fn(weight=weights, reduction="mean")
            logits = self.model(data_met, data_prot).squeeze()
            output = loss(logits, target)
            output.backward()
            if x % batch_acc == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
            elif x == len(dataloader)-1:
                self.optimizer.zero_grad()
            try:
                roc = roc_auc_score(target.cpu().numpy(),
                                    logits.detach().cpu().numpy())
                rocs.append(roc)
            except ValueError:
                pass
            loss_instance = output.data.cpu().numpy()
            train_losses.append(loss_instance)
            print(progress_bar(x+1, len(dataloader)), end="\r")
        return np.mean(train_losses), np.mean(np.array(rocs))

    def _torch_test(self, dataloader, weight_neg=0.9, return_preds=False):
        # Set evaluation mode for encoder and decoder
        self.model.eval()
        test_losses = []
        rocs = []
        preds = []
        targets = []
        with torch.no_grad():  # No need to track the gradients
            for x, batch in enumerate(dataloader):
                data_prot = batch[0].float().to(self.device)
                data_met = batch[1].float().to(self.device)
                weights = batch[2].float()
                target = weights.round()
                weights[weights == 0] = weight_neg
                weights = weights.mul(batch[3].squeeze())
                weights = weights.to(self.device)
                target = target.to(self.device)
                loss = self.loss_fn(weight=weights, reduction="mean")
                logits = self.model(data_met, data_prot).squeeze()
                output = loss(logits, target)
                loss_instance = output.data.cpu().numpy()
                test_losses.append(loss_instance)
                try:
                    roc = roc_auc_score(
                        target.cpu().numpy(), logits.detach().cpu().numpy())
                    rocs.append(roc)
                except ValueError:
                    pass
                if return_preds:
                    targets.append(target.cpu().numpy())
                    prediction = logits.cpu()
                    preds.append(prediction)
            if return_preds:
                preds = torch.concat(preds)
                preds = torch.sigmoid(preds)
                return np.mean(test_losses), np.mean(np.array(rocs)), preds.numpy(), np.concatenate(targets)
            else:
                return np.mean(test_losses), np.mean(np.array(rocs))

    def _torch_predict(self, pred_dataloader):
        self.model.eval()
        preds = []
        with torch.no_grad():  # No need to track the gradients
            for x, batch in enumerate(pred_dataloader):
                data_prot = batch[0].float().to(self.device)
                data_met = batch[1].float().to(self.device)
                logits = self.model(data_met, data_prot).squeeze()
                prediction = logits.cpu()
                if len(prediction.shape) == 0:
                    prediction = prediction[None]
                preds.append(prediction)
            preds = torch.concat(preds)
            preds = torch.sigmoid(preds)
            return preds.numpy()

    def _get_protein_weights(self, interactions, min_scale=0.5):
        interactions = self.df_interactions
        protein_counts = interactions["protein"].value_counts()
        protein_array = protein_counts.to_numpy()
        emms = MinMaxScaler([0.5, 1])
        log_freqs = np.log10((1/(protein_array/protein_array.sum())))
        scaled = (emms.fit_transform(log_freqs[:, None])
                  .astype(np.float32).squeeze())
        protein_weights = pd.DataFrame(
            [protein_counts.index, scaled]).transpose()

        protein_weights.columns = ["protein", "weight"]
        protein_weights = protein_weights.astype(
            {"weight": np.float32}).set_index("protein")
        return protein_weights

    def _process_external_data(self, df_interactions):
        pass

    def _add_negative_sample(self, df, reference_df, ratio=1):
        size = int(df.shape[0] * ratio)
        mets = df["chemical"]
        prots = df["protein"].unique()
        met_sample = np.random.choice(mets, size=size)
        prot_sample = np.random.choice(prots, size=size)
        random_sample = pd.DataFrame(
            {"chemical": met_sample, "protein": prot_sample, "combined_score": np.zeros([size])})
        repeated = reference_df.merge(
            random_sample.reset_index(), how='inner', on=["chemical", "protein"])
        to_drop = repeated["index"].to_numpy()
        random_sample = random_sample.drop(to_drop, axis=0)
        return pd.concat([df, random_sample], axis=0).reset_index(drop=True)

    def train(self, epochs, optimize_hyperparameters=False):
        if optimize_hyperparameters:
            raise NotImplementedError
            self.optimize_hyperparameters()
        self._train_on_data(epochs)

    def _train_on_data(self, epochs, neg_weight=0.9):
        self.diz_loss = {}
        test_dataloader = self.get_dataloader(
            self._test_split, self.df_interactions)
        for epoch in range(0, epochs):
            train_dataloader = self.get_dataloader(
                self._train_split, self.df_interactions)
            train_loss, train_roc = self._torch_train(train_dataloader)
            test_loss, test_roc = self._torch_test(test_dataloader)
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}\t train roc {} \t val roc {}'.format(
                epoch + 1, epochs, train_loss, test_loss, train_roc, test_roc))
            self.diz_loss['train_loss'].append(train_loss)
            self.diz_loss['val_loss'].append(test_loss)
            self.diz_loss['train_roc'].append(train_roc)
            self.diz_loss['val_roc'].append(test_roc)

    def load_trained(self, path=None):
        if path is None:
            self.model.load_state_dict(torch.load(self.file_path))
            print("Model loaded successfully")
        else:
            self.model.load_state_dict(torch.load(path))
            print("Model loaded successfully")

    def save_trained(self, path):
        pass

    def test_metrics(self, data=None, neg_weight=0.9):
        if data is None:
            dataloader = self.get_dataloader(
                self._test_split, self.df_interactions)
        else:
            dataloader = self.get_dataloader(
                data, self.df_interactions)
        return self._torch_test(dataloader, weight_neg=neg_weight)

    def test_predict(self, data=None, add_negative=True, neg_weight=0.9):
        if data is None:
            dataloader = self.get_dataloader(
                self._test_split, self.df_interactions, add_negative_sample=add_negative)
        else:
            dataloader = self.get_dataloader(
                data, self.df_interactions, add_negative_sample=add_negative)
        return self._torch_test(dataloader, weight_neg=neg_weight, return_preds=True)

    def _pairwise_predict(self, metabolites: np.array, proteins: np.array) -> pd.DataFrame:
        length = metabolites.shape[0]
        scores = np.zeros(length)
        predictions_df = (pd.DataFrame({"chemical": metabolites,
                                       "protein": proteins,
                                        "combined_score": scores})
                          .astype({"chemical": int}))
        loader = self.get_dataloader_pred(predictions_df)
        preds = self._torch_predict(loader)
        predictions_df = predictions_df.assign(combined_score=preds)
        return predictions_df

    def get_compounds(self, which="train"):
        assert(which in ["test", "train", "both"])
        if which == "train":
            prots = self._train_split["protein"].unique()
            mets = self._train_split["chemical"].unique()
        if which == "test":
            prots = self._test_split["protein"].unique()
            mets = self._test_split["chemical"].unique()
        if which == "both":
            prots_train = self._train_split["protein"].unique()
            mets_train = self._train_split["chemical"].unique()
            prots_test = self._test_split["protein"].unique()
            mets_test = self._test_split["chemical"].unique()
            prots = np.union1d(prots_train, prots_test)
            mets = np.union1d(mets_train, mets_test)
        return mets, prots

    def _crosswise_predict(self, metabolites: np.array, proteins: np.array) -> pd.DataFrame:
        metabolites = np.unique(metabolites)
        proteins = np.unique(proteins)
        predictions_df = (pd.DataFrame({"chemical": metabolites})
                          .astype({"chemical": int})
                          .merge(pd.DataFrame({"protein": proteins}), how="cross")
                          .assign(combined_score=0))
        loader = self.get_dataloader_pred(predictions_df)
        preds = self._torch_predict(loader)
        predictions_df = predictions_df.assign(combined_score=preds)
        return predictions_df
