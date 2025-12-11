# CS 6120 Final Project: Deep Learning Humor Prediction

Authors: Yuang Chen, Hao Ding, Abulajiang Abulamu

 Project Overview

This project explores various Natural Language Processing (NLP) models to predict the "funniness" of edited news headlines. We treat this as a regression task, predicting a continuous score (meanGrade) representing the humor intensity.

We implemented and compared the performance of the following models:
1.   TF-IDF + Linear Regression
2.   BiLSTM with GloVe Embeddings
3.   BERT (bert-base-uncased) and RoBERTa (roberta-base)

## Repository Structure

```text
.
├── data/                                   # Contains initial raw datasets
│   ├── train.csv
│   └── dev.csv
├── sub/                                    # Contains the pre-split train/test sets used in our final results
│   ├── new_train.csv
│   └── new_test.csv
├── Graph_csv/                              # Scripts for visualization and CSVs for plotting
│   ├── Overall Model Comparison (MAE)(1).py
│   ├── plot_epoch_mae(1).py
│   └── ...
├── new_train_linear_regression_baseline(3).py  # Baseline model & Data Splitter
├── _new_bilstm_glove_regression(3).py          # BiLSTM model training script
└── new_bert_roberta_training.py                # BERT/RoBERTa fine-tuning script
```

## Prerequisites & Installation

This project requires **Python 3.8+** and the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `datasets` (Hugging Face)
- `joblib`

You can install the dependencies via pip:

```bash
pip install pandas numpy scikit-learn matplotlib torch transformers datasets joblib
```

### Additional Setup for BiLSTM (GloVe Embeddings)
The BiLSTM model requires pre-trained GloVe embeddings.
1. Download `glove.6B.zip` from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/).
2. Unzip the file.
3. Create a folder named `embeddings/` in the root directory.
4. Place `glove.6B.100d.txt` inside `embeddings/`.

---

## Data Setup

The scripts are configured to read data from the `data/` directory. You have two options to set up the data:

### Option 1: Use Provided Split 
We have provided the exact training and testing splits used in our experiments in the `sub/` folder.
1. Copy the files from `sub/` to `data/`:
   - Copy `sub/new_train.csv` -> `data/new_train.csv`
   - Copy `sub/new_test.csv` -> `data/new_test.csv`

### Option 2: Generate from Raw Data
If you prefer to generate the splits from scratch using the original dataset:
1. Ensure `data/train.csv` exists.
2. Run the linear regression baseline script (see Step 1 below), which will automatically generate `data/new_train.csv` and `data/new_test.csv`.

---

## Instructions to Run

### 1. Baseline Model (TF-IDF + Linear Regression)
Run this script first if you need to generate the data splits from `data/train.csv`. It trains a Linear Regression model using TF-IDF features.

```bash
python "new_train_linear_regression_baseline(3).py"
```
*Output:* Prints MAE/RMSE scores and saves the model to `.joblib`.

### 2. BiLSTM Model
Trains a Bidirectional LSTM using GloVe word embeddings.
*Note: Ensure you have placed `glove.6B.100d.txt` in the `embeddings/` folder as described in Prerequisites.*

```bash
python "_new_bilstm_glove_regression(3).py"
```
*Output:* Trains for 10 epochs, saves the best model, and prints validation metrics.

### 3. BERT & RoBERTa Models
Performs a grid search for hyperparameters, fine-tunes BERT and RoBERTa models, and generates predictions using the best performing model.

```bash
python new_bert_roberta_training.py
```
*Output:* Selected best model (RoBERTa), re-trains it, and saves predictions to `dev_predictions.csv`.

### 4. Visualization & Analysis
To generate the performance comparison plots included in the report:

Epoch-wise MAE Comparison:**
```bash
python "Graph_csv/plot_epoch_mae(1).py"
```

Overall Model Comparison:**
```bash
python "Graph_csv/Overall Model Comparison (MAE)(1).py"
```

