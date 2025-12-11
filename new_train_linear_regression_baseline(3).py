import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

# 1. Load the original data
df = pd.read_csv('data/train.csv')

# 2. Perform the split (Fixed Random State = 42)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Save to disk
train_df.to_csv('data/new_train.csv', index=False)
test_df.to_csv('data/new_test.csv', index=False)

print(f"Split completed")
print(f"New Train size: {len(train_df)}")


# Paths
TRAIN_PATH = "data/new_train.csv"
DEV_PATH   = "data/new_test.csv"  # used as dev/validation set

TEXT_COL  = "text"        # full edited headline
LABEL_COL = "meanGrade"

# Load and preprocess TRAIN
print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)
print("Columns in new_train.csv:", train_df.columns.tolist())

# Build full edited headline: replace <...> in original with the edited word
train_df["text"] = train_df.apply(
    lambda x: re.sub(r"<.+?>", x["edit"], x["original"]),
    axis=1
)


train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL])

X_train = train_df[TEXT_COL].astype(str).values
y_train = train_df[LABEL_COL].astype(float).values

print(f"Train samples: {len(X_train)}")

# Load and preprocess DEV
print("Loading dev data from new_test.csv...")
dev_df = pd.read_csv(DEV_PATH)
print("Columns in new_test.csv:", dev_df.columns.tolist())

dev_df["text"] = dev_df.apply(
    lambda x: re.sub(r"<.+?>", x["edit"], x["original"]),
    axis=1
)

dev_df = dev_df.dropna(subset=[TEXT_COL, LABEL_COL])

X_dev = dev_df[TEXT_COL].astype(str).values
y_dev = dev_df[LABEL_COL].astype(float).values

print(f"Dev samples: {len(X_dev)}")

# Build TF-IDF + Linear Regression pipeline
model = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 2),    # unigrams + bigrams
        max_features=50000,    # cap vocabulary size
    ),
    LinearRegression()
)

# Train the model
print("Training TF-IDF + Linear Regression model...")
model.fit(X_train, y_train)

# Evaluate on the dev set
print("Evaluating on dev (new_test.csv)...")
y_pred = model.predict(X_dev)

mae = mean_absolute_error(y_dev, y_pred)
rmse = np.sqrt(mean_squared_error(y_dev, y_pred))


print(f"Dev MAE  = {mae:.4f}")
print(f"Dev RMSE = {rmse:.4f}")


# Save the model

joblib.dump(model, "linear_regression_tfidf.joblib")
print("Model saved to linear_regression_tfidf.joblib")



# Histogram of funniness scores in the training set
plt.figure(figsize=(6, 4))
plt.hist(y_train, bins=20, edgecolor="black")
plt.xlabel("Funniness score (0–3)")
plt.ylabel("Count")
plt.title("Distribution of funniness scores (train set)")
plt.tight_layout()
plt.savefig("plot_funniness_distribution.png", dpi=300)
print("Saved: plot_funniness_distribution.png")

# Scatter plot: true vs predicted scores on dev set
plt.figure(figsize=(6, 6))
plt.scatter(y_dev, y_pred, alpha=0.5)
# Reference line y = x
min_val = min(y_dev.min(), y_pred.min())
max_val = max(y_dev.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("True funniness score")
plt.ylabel("Predicted funniness score")
plt.title("True vs Predicted scores (dev set)")
plt.tight_layout()
plt.savefig("plot_true_vs_predicted.png", dpi=300)
print("Saved: plot_true_vs_predicted.png")

# Histogram of prediction errors (residuals)
residuals = y_pred - y_dev

plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=20, edgecolor="black")
plt.xlabel("Prediction error (predicted - true)")
plt.ylabel("Count")
plt.title("Distribution of prediction errors (dev set)")
plt.tight_layout()
plt.savefig("plot_residuals_distribution.png", dpi=300)
print("Saved: plot_residuals_distribution.png")
