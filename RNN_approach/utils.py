import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from datetime import datetime
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import io

def now_date():
    return datetime.now().strftime("%d%m%Y_%H%M")

class SensorSequenceDataset(Dataset):
    def __init__(self, df, exp_name_to_label, label_encoder, mean, std):
        self.exp_name_to_label = exp_name_to_label
        self.label_encoder = label_encoder
        self.mean = mean
        self.std = std

        self.feature_cols = [col for col in df.columns if col not in ["label", "exp_name"]]

        # Raggruppa per esperimento
        self.groups = df.groupby("exp_name")
        self.exp_names = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.exp_names)

    def __getitem__(self, idx):
        exp_name = self.exp_names[idx]
        group = self.groups.get_group(exp_name)

        # Estrai e normalizza le feature
        features = group[self.feature_cols].values.astype("float32")
        features = (features - self.mean) / self.std
        features = torch.tensor(features, dtype=torch.float32)  # shape: (M, 16)

        # Etichetta associata all'esperimento
        label = self.label_encoder.transform([self.exp_name_to_label[exp_name]])[0]

        return features, label, exp_name


class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim=16,
        hidden_dim=64,
        output_dim=17,
        rnn_type="GRU",  # "RNN", "LSTM", "GRU"
        lr=1e-4,
        weight_decay=1e-5,
        dropout=0.2,
        num_layers=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        rnn_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]

        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_preds = []
        self.test_targets = []

    def forward(self, x, lengths):
        # Packing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Prendi il last valid hidden state per ogni sequenza
        idx = (lengths - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        last_hidden = out.gather(1, idx).squeeze(1)

        return self.fc(self.dropout(last_hidden))

    def training_step(self, batch, batch_idx):
        x, lengths, y = batch
        logits = self(x, lengths)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, y = batch
        logits = self(x, lengths)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, lengths, y = batch
        logits = self(x, lengths)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)

        cm = confusion_matrix(targets, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.show()
        plt.close(fig)

        # Log other metrics if needed
        self.log("final_test_acc", (preds == targets).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

def collate_fn_padded(batch):
    sequences, labels = zip(*[(x[0], x[1]) for x in batch])
    
    # Lista di tensori (batch_size, seq_len, input_dim)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    
    # Padding: (batch_size, max_len, input_dim)
    padded_sequences = pad_sequence(sequences, batch_first=True)  # zero-padding
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_sequences, lengths, labels
