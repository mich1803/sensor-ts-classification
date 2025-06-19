# -*- coding: utf-8 -*-

"""

This module contains the definition of the dataset class, various classifiers, and utility functions for training and evaluating models on time windowed sensor data.

It includes the following classes:
- SensorDataset: A PyTorch dataset class for handling time windows of sensor data.
- FilteredSensorDataset: A subclass of SensorDataset that filters the dataset based on specified labels.
- ColumnSelectiveEncoding: A PyTorch module for column-selective encoding of sensor data.
- CSEncoderMLPClassifier: A PyTorch Lightning module for a classifier using column-selective encoding and MLP.
- EncoderMLPClassifier: A PyTorch Lightning module for a classifier using an encoder and MLP.
- MLPClassifier: A PyTorch Lightning module for a simple MLP classifier.
- CNNClassifier: A PyTorch Lightning module for a CNN classifier.

It also includes utility functions for:
- now_date: A function to get the current date and time in a specific format.
- get_stats: A function to compute confusion matrix and experiment-level statistics using a DataLoader.
- compute_experiment_performance: A function to compute experiment-level accuracy and confusion matrix.
- plot_confusion_matrices: A function to plot confusion matrices for train, validation, and test sets.

"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict

def now_date():
    return datetime.now().strftime("%d%m%Y_%H%M")

class SensorDataset(Dataset):
    def __init__(self, df, exp_name_to_label, label_encoder, mean, std, N=10, overlap=None):
        self.df = df.reset_index(drop=True)
        self.N = N
        self.overlap = N - 1 if overlap is None else overlap
        self.step = N - self.overlap
        self.exp_name_to_label = exp_name_to_label
        self.label_encoder = label_encoder
        self.mean = mean
        self.std = std

        self.feature_cols = [col for col in df.columns if col not in ["label", "exp_name"]]

        # Precompute valid starting indices
        self.valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self):
        valid = []
        i = 0
        while i + self.N <= len(self.df):
            window = self.df.iloc[i:i + self.N]
            if len(window["exp_name"].unique()) == 1:
                valid.append(i)
                i += self.step
            else:
                i += 1
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        window = self.df.iloc[start_idx:start_idx + self.N]

        features = window[self.feature_cols].values.astype(np.float32)
        features = (features - self.mean) / self.std  # <-- fix qui
        features = torch.tensor(features, dtype=torch.float32)

        exp_name = window["exp_name"].iloc[0]
        label = self.label_encoder.transform([self.exp_name_to_label[exp_name]])[0]

        return features, label, exp_name
    
    def plot_sample(self, idx=None):
        """Plots a sample feature matrix from the dataset."""
        if idx is None:
            # Random index if not provided
            idx = np.random.randint(0, len(self))

        # Get the corresponding sample
        features, _, _ = self[idx]

        # Plotting the matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(features.numpy(), cmap='viridis', aspect='auto') 
        plt.colorbar(label="Feature Value")

        # Labeling the axes
        plt.title(f"Sample {idx}")
        plt.xlabel("Features (16)")
        plt.ylabel("Time steps (N)")

        plt.show()

class FilteredSensorDataset(SensorDataset):
    def __init__(self, df, exp_name_to_label, mean, std, labels_to_include, N=10, overlap=None):
        """
        labels_to_include: dict mapping label names (e.g., 'ETHANOL') to class IDs (e.g., 0 or 1)
        """
        self.labels_to_include = labels_to_include

        # Only keep exp_names that map to the labels we want
        included_exp_names = [
            exp_name for exp_name, label in exp_name_to_label.items()
            if label in labels_to_include
        ]

        # Filter the dataframe accordingly
        filtered_df = df[df["exp_name"].isin(included_exp_names)].reset_index(drop=True)

        # Save label mapping dict and exp_name->label mapping
        self.exp_name_to_label = exp_name_to_label
        self.label_map = labels_to_include

        # Call parent constructor (label_encoder is not needed anymore, pass None)
        super().__init__(filtered_df, exp_name_to_label, label_encoder=None, mean=mean, std=std, N=N, overlap=overlap)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        window = self.df.iloc[start_idx:start_idx + self.N]

        features = window[self.feature_cols].values.astype(np.float32)
        features = (features - self.mean) / self.std
        features = torch.tensor(features, dtype=torch.float32)

        exp_name = window["exp_name"].iloc[0]
        raw_label = self.exp_name_to_label[exp_name]
        label = self.label_map[raw_label]  # label is now a clean 0/1 integer

        return features, label, exp_name

    

class ColumnSelectiveEncoding(nn.Module):
    def __init__(self, num_features, window_size, m, k):
        super().__init__()
        self.num_features = num_features  # 16
        self.window_size = window_size    # N
        self.m = m  # number of column-specific nodes per feature
        self.k = k  # number of fully connected nodes

        # Column-specific weights: one linear layer per feature column (shared across timesteps)
        self.column_linears = nn.ModuleList([
            nn.Linear(window_size, m) for _ in range(num_features)
        ])

        # Fully connected part: a standard FC from the full (N × num_features) input
        self.full_linear = nn.Linear(num_features * window_size, k)

    def forward(self, x):
        # x shape: (batch_size, N, num_features)
        x = x.permute(0, 2, 1)  # -> (batch_size, num_features, N)

        # Apply column-specific encodings
        column_outputs = []
        for i in range(self.num_features):
            column = x[:, i, :]              # (batch_size, N)
            out = self.column_linears[i](column)  # (batch_size, m)
            column_outputs.append(out)

        # Concatenate all column-specific outputs: (batch_size, m * num_features)
        col_encoded = torch.cat(column_outputs, dim=1)

        # Fully connected part
        flat_input = x.permute(0, 2, 1).reshape(x.size(0), -1)  # (batch_size, N * num_features)
        full_encoded = self.full_linear(flat_input)            # (batch_size, k)

        # Final concatenated encoding
        encoded = torch.cat([col_encoded, full_encoded], dim=1)  # (batch_size, m*features + k)
        return encoded


class CSEncoderMLPClassifier(pl.LightningModule):
    def __init__(self, num_features, window_size, m, k, hidden_dim, num_classes,
                 lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        encoded_dim = num_features * m + k
        self.encoder = ColumnSelectiveEncoding(num_features, window_size, m, k)
        self.mlp = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.encoder(x)
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("train_acc", acc, prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("val_acc", acc, prog_bar=True, batch_size=len(x))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("test_acc", acc, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )


class EncoderMLPClassifier(pl.LightningModule):
    def __init__(self, num_features, window_size, latent_dim, hidden_dim, num_classes,
                 lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        input_dim = num_features * window_size

        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("train_acc", acc, prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("val_acc", acc, prog_bar=True, batch_size=len(x))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("test_acc", acc, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

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

    def forward(self, x):
        # x shape: (batch_size, sequence_len, input_dim)
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]  # Take the last time step
        out = self.fc(self.dropout(last_hidden))
        return out

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
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

        self.log("final_test_acc", (preds == targets).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


class MLPClassifier(pl.LightningModule):
    def __init__(self, num_features, window_size, hidden_dim, num_classes,
                 lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        input_dim = num_features * window_size

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("train_acc", acc, prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("val_acc", acc, prog_bar=True, batch_size=len(x))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("test_acc", acc, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

class CNN2DClassifier(pl.LightningModule):
    def __init__(self, num_features, window_size, hidden_dim, num_classes, new_channels=4,
                 kernel_size = 2, lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Input shape: (batch_size, N, num_features)
        # Reshape to: (batch_size, 1, N, num_features) for Conv2d
        self.cnn = nn.Sequential(
            nn.Conv2d(1, new_channels, kernel_size=2),  # out: (batch_size, 16, N-1, 15)
            nn.ReLU(),
            nn.Dropout(dropout)
            )

        # Compute flattened size after convolutions
        dummy_input = torch.zeros(1, 1, window_size, num_features)
        conv_output_dim = self.cnn(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, N, 16)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("train_acc", acc, prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("val_acc", acc, prog_bar=True, batch_size=len(x))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("test_acc", acc, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )


class CNN1DClassifier(pl.LightningModule):
    def __init__(self, num_features, window_size, hidden_dim, num_classes,
                 out_channels=4, kernel_size=2, lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Input shape: (batch_size, window_size, num_features)
        # Conv1D expects: (batch_size, num_features, window_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Calcolo della dimensione in output della CNN per determinare l'input del Linear
        dummy_input = torch.zeros(1, num_features, window_size)
        conv_output_dim = self.cnn(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: (B, window_size, num_features) -> (B, num_features, window_size)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("train_acc", acc, prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("val_acc", acc, prog_bar=True, batch_size=len(x))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=len(x))
        self.log("test_acc", acc, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )



def get_stats(model_path, dataloader, num_classes, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads a model and computes confusion matrix and experiment-level statistics using a DataLoader.
    
    Parameters:
    - model_path: path to the model checkpoint
    - dataloader: DataLoader for the dataset
    - num_classes: number of classes in the dataset
    - device: "cuda" or "cpu"
    
    Returns:
    - cm: sample-level confusion matrix (num_classes x num_classes)
    - exp_stats: dict: {exp_name -> {"ground_truth_label", "predictions": {label: count}}}
    """
    # Load model based on prefix
    
    try:
        if os.path.basename(model_path).split("_")[0].startswith("binary"):
            model_type = os.path.basename(model_path).split("_")[1]
        else:
            model_type = os.path.basename(model_path).split("_")[0]
        if model_type == "MLP":
            model = MLPClassifier.load_from_checkpoint(model_path)
        elif model_type == "EMLP":
            model = EncoderMLPClassifier.load_from_checkpoint(model_path)
        elif model_type == "CSEMLP":
            model = CSEncoderMLPClassifier.load_from_checkpoint(model_path)
        elif model_type == "CNN2D":
            model = CNN2DClassifier.load_from_checkpoint(model_path)
        elif model_type == "CNN1D":
            model = CNN1DClassifier.load_from_checkpoint(model_path)
        elif model_type == "RNN":
            model = RNNClassifier.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")
    
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    exp_stats = {}

    with torch.no_grad():
        for x_batch, y_batch, exp_names in tqdm(dataloader):
            x_batch = x_batch.to(device)
            logits = model(x_batch.contiguous())
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y_batch.numpy()

            y_true.extend(labels)
            y_pred.extend(preds)

            for exp_name, label, pred in zip(exp_names, labels, preds):
                if exp_name not in exp_stats:
                    exp_stats[exp_name] = {
                        "ground_truth_label": label.item() if hasattr(label, "item") else label,
                        "predictions": defaultdict(int)
                    }
                exp_stats[exp_name]["predictions"][pred] += 1

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm, exp_stats



def compute_experiment_performance(exp_stats, num_classes=None):
    """
    Computes experiment-level accuracy using majority vote, and returns an experiment-level confusion matrix.
    
    Parameters:
    - exp_stats: dictionary containing for each exp_name:
        - 'ground_truth_label': the true label
        - 'predictions': a dict of predicted label counts
    - num_classes: total number of classes (optional; inferred if not provided)
    
    Returns:
    - performance: float, ratio of correct majority-vote predictions
    - majority_votes: dict mapping exp_name to (majority_label, correct: bool)
    - confmat: 2D numpy array of shape (num_classes, num_classes), experiment-level confusion matrix
    """
    correct = 0
    total = 0
    majority_votes = []
    all_true = []
    all_pred = []

    for exp_name, stats in exp_stats.items():
        preds = stats["predictions"]
        true_label = stats["ground_truth_label"]

        # Majority vote
        majority_label = max(preds.items(), key=lambda x: x[1])[0]
        is_correct = (majority_label == true_label)

        if is_correct:
            correct += 1
        total += 1

        majority_votes.append({
            "exp_name": exp_name,
            "majority_label": majority_label,
            "ground_truth": true_label,
            "correct": is_correct
        })

        all_true.append(true_label)
        all_pred.append(majority_label)

    performance = correct / total if total > 0 else 0.0

    # Compute experiment-level confusion matrix
    if num_classes is None:
        num_classes = max(max(all_true), max(all_pred)) + 1

    confmat = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(all_true, all_pred):
        confmat[t][p] += 1

    return performance, majority_votes, confmat


def get_stats_f1(model_path, dataloader, num_classes, device="cuda" if torch.cuda.is_available() else "cpu"):

    # Model loading (same as before)
    try:
        if os.path.basename(model_path).split("_")[0].startswith("binary"):
            model_type = os.path.basename(model_path).split("_")[1]
        else:
            model_type = os.path.basename(model_path).split("_")[0]
        if model_type == "MLP":
            model = MLPClassifier.load_from_checkpoint(model_path)
        elif model_type == "EMLP":
            model = EncoderMLPClassifier.load_from_checkpoint(model_path)
        elif model_type == "CSEMLP":
            model = CSEncoderMLPClassifier.load_from_checkpoint(model_path)
        elif model_type == "CNN":
            model = CNNClassifier.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")
    
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    exp_stats = {}

    weights = [2.5, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

    with torch.no_grad():
        for x_batch, y_batch, exp_names in tqdm(dataloader):
            x_batch = x_batch.to(device)
            logits = model(x_batch.contiguous())
            probs = torch.softmax(logits, dim=1)

            topk_vals, topk_indices = probs.topk(k=10, dim=1)  # (B, 10)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y_batch.numpy()

            y_true.extend(labels)
            y_pred.extend(preds)

            for exp_name, label, top_indices in zip(exp_names, labels, topk_indices.cpu().numpy()):
                if exp_name not in exp_stats:
                    exp_stats[exp_name] = {
                        "ground_truth_label": int(label),
                        "weighted_predictions": defaultdict(float)
                    }
                for rank, pred_label in enumerate(top_indices):
                    weight = weights[rank] if rank < len(weights) else 0.0
                    exp_stats[exp_name]["weighted_predictions"][int(pred_label)] += weight

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm, exp_stats


def compute_experiment_performance_f1(exp_stats, num_classes=None):

    correct = 0
    total = 0
    majority_votes = []
    all_true = []
    all_pred = []

    for exp_name, stats in exp_stats.items():
        preds = stats["weighted_predictions"]
        true_label = stats["ground_truth_label"]

        # Soft majority vote
        majority_label = max(preds.items(), key=lambda x: x[1])[0]
        is_correct = (majority_label == true_label)

        if is_correct:
            correct += 1
        total += 1

        majority_votes.append({
            "exp_name": exp_name,
            "majority_label": majority_label,
            "ground_truth": true_label,
            "correct": is_correct
        })

        all_true.append(true_label)
        all_pred.append(majority_label)

    performance = correct / total if total > 0 else 0.0

    if num_classes is None:
        num_classes = max(max(all_true), max(all_pred)) + 1

    confmat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_true, all_pred):
        confmat[t][p] += 1

    return performance, majority_votes, confmat



def plot_confusion_matrices(
    train_cm, val_cm, test_cm, 
    suptitle, 
    class_names=None, 
    cmap="Blues", 
    savepath=None, 
    cm_type="Sample"
):
    """
    Plots confusion matrices for train, validation, and test sets.
    """
    norm_cms = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in [train_cm, val_cm, test_cm]]

    fig, axes = plt.subplots(
        1, 3, figsize=(22, 7),
        constrained_layout=True,
        gridspec_kw={'wspace': 0.1}  # <-- more horizontal space between plots
    )
    fig.suptitle(suptitle, fontsize=18)

    titles = ['Train Set', 'Validation Set', 'Test Set']
    original_cms = [train_cm, val_cm, test_cm]

    for i, ax in enumerate(axes):
        sns.heatmap(
            norm_cms[i],
            ax=ax,
            annot=False,
            cmap=cmap,
            xticklabels=class_names if class_names is not None else True,
            yticklabels=class_names if class_names is not None else True,
            cbar=i == 2,
            cbar_kws={"shrink": 0.75, "aspect": 30}
        )
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
        ax.tick_params(axis='y', labelrotation=0, labelsize=7)

        row_sums = original_cms[i].sum(axis=1)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel(f'Total {cm_type}s', color='gray', fontsize=10)
        ax2.set_yticks(np.arange(len(row_sums)) + 0.5)
        ax2.set_yticklabels([str(int(s)) for s in row_sums], fontsize=7, color='gray')
        ax2.tick_params(axis='y', colors='gray')

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()