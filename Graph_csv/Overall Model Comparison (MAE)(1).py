import matplotlib.pyplot as plt

# Final MAE values on new_test.csv (used as dev)
models = [
    "TF-IDF + Linear\nRegression",
    "BiLSTM +\nGloVe",
    "BERT-base",
    "RoBERTa-base"
]

mae_values = [
    0.5200,    # TF-IDF + Linear Regression on new_test
    0.4477,    # BiLSTM + GloVe best Dev MAE (epoch 6)
    0.427675,  # best BERT-base MAE from leaderboard (LR = 3e-5)
    0.418190   # best RoBERTa-base MAE from leaderboard (LR = 3e-5, winner)
]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, mae_values)

plt.ylabel("MAE (lower is better)")
plt.title("Model Comparison on new_test (Validation Set)")

# Add value labels on top of bars
for bar, val in zip(bars, mae_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.3f}",
        ha="center",
        va="bottom"
    )

# Zoom y-axis to highlight differences
plt.ylim(0.41, 0.54)

plt.tight_layout()
plt.savefig("model_mae_comparison.png", dpi=300)
print("Saved: model_mae_comparison.png")
# plt.show()  # uncomment if you want to display the figure interactively
