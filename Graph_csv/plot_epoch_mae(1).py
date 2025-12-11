import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]


# RoBERTa-base MAE curves

roberta_lr2 = [0.442853, 0.422851, 0.460887, 0.466578, 0.459415]
roberta_lr3 = [0.431919, 0.418190, 0.448127, 0.462799, 0.465995]
roberta_lr5 = [0.446610, 0.430457, 0.456732, 0.479069, 0.468464]


# BERT-base MAE curves

bert_lr2 = [0.441799, 0.429795, 0.442627, 0.458567, 0.464400]
bert_lr3 = [0.444284, 0.427675, 0.439198, 0.461594, 0.464954]
bert_lr5 = [0.444933, 0.434470, 0.445023, 0.463473, 0.462534]

plt.figure(figsize=(10, 6))

# Plot RoBERTa
plt.plot(epochs, roberta_lr2, marker='o', label="RoBERTa LR=2e-5")
plt.plot(epochs, roberta_lr3, marker='o', label="RoBERTa LR=3e-5 (best)")
plt.plot(epochs, roberta_lr5, marker='o', label="RoBERTa LR=5e-5")

# Plot BERT
plt.plot(epochs, bert_lr2, marker='o', linestyle='--', label="BERT LR=2e-5")
plt.plot(epochs, bert_lr3, marker='o', linestyle='--', label="BERT LR=3e-5")
plt.plot(epochs, bert_lr5, marker='o', linestyle='--', label="BERT LR=5e-5")

plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Epoch-wise MAE Comparison Across Models and Learning Rates")
plt.xticks(epochs)
plt.ylim(0.40, 0.50)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("epoch_mae_comparison.png", dpi=300)
print("Saved: epoch_mae_comparison.png")
# plt.show()
