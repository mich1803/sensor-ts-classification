# 🧪 Sensor Signal Classification – ML for Real-World Time Series

This repository explores **machine learning approaches for classifying sensor-based time series data** collected from real-world experiments involving different chemical substances.

## 📌 Task Overview

The dataset consists of multiple experiments, each represented as a multivariate time series with **16 sensor measurements over time** and labeled with one of **17 target substances**. Each experiment varies in length and quality.

The classification task is to predict the substance present in a given experiment, using different preprocessing and modeling strategies:

- **Time Window Approach**: Split experiments into fixed-length time windows (e.g. 100×16) to capture short-term dynamics.
- **Instant Time Approach**: Use each individual time point (1×16 vector) as an independent sample.
- **RBF Approximation Approach**: Approximate the entire time series with Radial Basis Functions (RBFs) and use this compact representation for classification.

## 📁 Project Structure

```bash 
sensor-ts-classification/
│
├── dataset/                    # Private
│
├── instant_approach/           # Models trained on individual time steps (1×16 vectors)
│   ├── preprocessing.ipynb     # Data filtering and preparation
│   ├── training.ipynb          # Model training notebook
│   ├── training_evaluating_binary.ipynb  # ETHANOL vs. ACETONE experiment
│   ├── evaluating.ipynb        # Test and inference
│   ├── explaining.ipynb        # Sample importance in time series
│   └── utils.py
│
├── time_windows_approach/      # Models trained on 100×16 time window slices
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   ├── training_evaluating_binary.ipynb
│   ├── evaluating.ipynb
│   ├── explaining.ipynb
│   └── utils.py
│
└── RBF_approach/               # Approximation of full time series using RBFs
    ├── preprocessing.ipynb
    ├── training.ipynb
    └── utils.py
```


## 🛠️ Implementation Notes

- All models are implemented using **PyTorch Lightning**.
- Each subdirectory is self-contained, with its own preprocessing and training notebooks.
- Preprocessing involves filtering for valid experiments (e.g. ≥200 rows) and substances with enough samples.
- Train/Val/Test splits are performed **per experiment**, preserving the sequential structure.

## 🧬 Modeling Highlights

- **Time Windows**: Includes MLPs, CNNs, and Encoders with/without column attention (CSE).
- **Instant Time**: Simpler MLP and encoder-based models to study classification without temporal context.
- **RBF Approach**: Fits basis functions to entire experiment signals and uses the resulting features for classification.
