
"""
This module contains utility functions and classes for training and evaluating machine learning models.
It includes:
- `SensorDataset`: A PyTorch Dataset class for loading sensor data.
- `EncoderMLPClassifier`: A PyTorch Lightning module for a multi-layer perceptron classifier.
- `MLPClassifier`: A PyTorch Lightning module for a multi-layer perceptron classifier.
- `get_stats`: A function to compute confusion matrix and experiment-level statistics.
- `compute_experiment_performance`: A function to compute experiment-level accuracy and confusion matrix.
- `plot_confusion_matrices`: A function to plot confusion matrices for train, validation, and test sets.
- `now_date`: A function to get the current date and time in a specific format.
"""

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

def now_date():
    return datetime.now().strftime("%d%m%Y_%H%M")

class SensorDataset(Dataset):
    def __init__(self, df, exp_name_to_label, label_encoder, mean, std):
        self.df = df.reset_index(drop=True)
        self.exp_name_to_label = exp_name_to_label
        self.label_encoder = label_encoder
        self.mean = mean
        self.std = std

        self.feature_cols = [col for col in df.columns if col not in ["label", "exp_name"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = row[self.feature_cols].values.astype(np.float32)
        features = (features - self.mean) / self.std
        features = torch.tensor(features, dtype=torch.float32)

        exp_name = row["exp_name"]
        label = self.label_encoder.transform([self.exp_name_to_label[exp_name]])[0]

        return features, label, exp_name

class EncoderMLPClassifier(pl.LightningModule):
    def __init__(self, num_features, latent_dim, hidden_dim, num_classes,
                 lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(num_features, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
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
    
class MLPClassifier(pl.LightningModule):
    def __init__(self, num_features, hidden_dim, num_classes,
                 lr=1e-3, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
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
    try:
        filename = os.path.basename(model_path)
        if filename.startswith("binary"):
            model_type = filename.split("_")[1]
        else:
            model_type = filename.split("_")[0]

        if model_type == "MLP":
            model = MLPClassifier.load_from_checkpoint(model_path)
        elif model_type == "EMLP":
            model = EncoderMLPClassifier.load_from_checkpoint(model_path)
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
            logits = model(x_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y_batch.numpy()

            y_true.extend(labels)
            y_pred.extend(preds)

            for exp_name, label, pred in zip(exp_names, labels, preds):
                if exp_name not in exp_stats:
                    exp_stats[exp_name] = {
                        "ground_truth_label": int(label),
                        "predictions": defaultdict(int)
                    }
                exp_stats[exp_name]["predictions"][int(pred)] += 1

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
    - majority_votes: list of dicts with exp_name, majority_label, ground_truth, correct
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