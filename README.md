# 🧪 Sensor Signal Classification – ML for Real-World Time Series

This repository explores **machine learning approaches for classifying multivariate time series data** collected from real-world sensor experiments involving different chemical substances.

## 📌 Problem Overview

Each experiment consists of a variable-length time series with **16 sensor channels**, capturing diverse signal behaviors under different environmental and chemical conditions. The task is to classify each experiment into one of **17 chemical classes**.

To tackle this challenge, the project investigates multiple modeling strategies with different levels of temporal abstraction:

- **🧩 Instant Time Approach**: Classify each individual time step as a standalone 1×16 feature vector.
- **📐 Time Window Approach**: Use fixed-length slices of 100×16 to incorporate short-term temporal dynamics.
- **🎯 RBF Approximation**: Compress entire time series into Radial Basis Function (RBF) coefficients and classify in a low-dimensional space.
- **🔁 Temporal Processing (RNN)**: Leverage recurrent neural architectures (GRU, LSTM) to process the entire sequence with variable length for global temporal encoding.

---

## 📁 Project Structure

```bash
sensor-ts-classification/
│
├── docs/
│   ├── report_ita.pdf                      # Final thesis in Italian 🇮🇹
│   └── report_eng.pdf                      # [English version of the thesis](docs/report_eng.pdf) 🇬🇧
│
├── instant_approach/                      # Models on isolated 1×16 time steps
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   ├── evaluating.ipynb
│   ├── explaining.ipynb
│   └── training_evaluating_binary.ipynb
│
├── time_windows_approach/                 # Models on 100×16 windows with attention/encoders
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   ├── evaluating.ipynb
│   ├── explaining.ipynb
│   └── training_evaluating_binary.ipynb
│
├── RBF_approach/                          # Time series compression with RBF coefficients
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   └── rbf_features.csv
│
├── RNN_approach/                          # Sequence-level classification with GRU/LSTM
│   ├── training.ipynb
│   └── utils.py
│
└── data_analysis.ipynb                     

```

---

## 🛠️ Implementation Notes

- Implemented in **PyTorch Lightning** for modular training, evaluation, and logging.
- Each approach directory contains independent preprocessing, training, and evaluation code.
- Data filtering excludes experiments with fewer than 200 time steps.
- Train/validation/test splits are done per-experiment to maintain time coherence and prevent data leakage.

---

## 📊 Modeling Summary

| Approach                | Input Shape        | Models Used                        |
|------------------------|--------------------|------------------------------------|
| Instant Time           | 1×16               | MLP, Encoder MLP                   |
| Time Window            | 100×16             | MLP, CNN, Encoder, CSE             |
| RBF Approximation      | RBF Coeffs (16×k)  | MLP, Random Forest                 |
| Temporal (RNN)         | Variable × 16      | GRU, LSTM                          |

---

## 📄 Read the Full Thesis

📘 [Read the full thesis in English (PDF)](docs/report_eng.pdf)


