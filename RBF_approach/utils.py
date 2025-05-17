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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    
class MLPClassifier(pl.LightningModule):
    def __init__(self, num_features, hidden_dims, num_classes, lr=1e-4, dropout=0.2, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        dims = [num_features] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, batch_size=y.size(0))
        self.log("train_acc", acc, prog_bar=True, batch_size=y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, batch_size=y.size(0))
        self.log("val_acc", acc, prog_bar=True, batch_size=y.size(0))

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
    
def plot_confusion_matrix(model, dataloader, title="Confusion Matrix", class_names=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Plots the confusion matrix of a model on a given dataloader.

    Parameters:
    - model: PyTorch or Lightning model
    - dataloader: DataLoader with (features, labels, exp_name) triplets
    - title: Title of the plot
    - class_names: List of class names for display (optional)
    - device: Device to run inference on
    """
    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()