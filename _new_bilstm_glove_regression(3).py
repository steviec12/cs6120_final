import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Config
TRAIN_PATH = "data/new_train.csv"
DEV_PATH   = "data/new_test.csv"

TEXT_COL   = "text"        # full edited headline
LABEL_COL  = "meanGrade"   # mean funniness score

GLOVE_PATH = "embeddings/glove.6B.100d.txt"
EMBED_DIM  = 100
HIDDEN_DIM = 256
NUM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.3
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3
MAX_LEN = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)
print("Columns in new_train.csv:", train_df.columns.tolist())

train_df["text"] = train_df.apply(
    lambda x: re.sub(r"<.+?>", x["edit"], x["original"]),
    axis=1
)

train_df = train_df.dropna(subset=[TEXT_COL, LABEL_COL])
X_train = train_df[TEXT_COL].astype(str).values
y_train = train_df[LABEL_COL].astype(float).values

print("Train samples:", len(X_train))

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

print("Dev samples:", len(X_dev))

# Build vocabulary from TRAIN only
def simple_tokenize(text):
    # very simple whitespace tokenizer
    return text.strip().split()

word_freq = {}
for sent in X_train:
    for token in simple_tokenize(sent.lower()):
        word_freq[token] = word_freq.get(token, 0) + 1

min_freq = 1
itos = ["<PAD>", "<UNK>"]
for w, f in word_freq.items():
    if f >= min_freq:
        itos.append(w)

stoi = {w: i for i, w in enumerate(itos)}
vocab_size = len(itos)
print("Vocab size:", vocab_size)

# Load GloVe embeddings
def load_glove_embeddings(glove_path, embed_dim, stoi):
    print("Loading GloVe embeddings from:", glove_path)
    embeddings = np.random.normal(scale=0.6, size=(len(stoi), embed_dim)).astype(np.float32)
    # PAD = 0 vector
    embeddings[0] = np.zeros(embed_dim, dtype=np.float32)

    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove_dict[word] = vec

    hit = 0
    for word, idx in stoi.items():
        if word in glove_dict:
            embeddings[idx] = glove_dict[word]
            hit += 1
    print(f"GloVe hits: {hit}/{len(stoi)}")
    return torch.tensor(embeddings)

embedding_matrix = load_glove_embeddings(GLOVE_PATH, EMBED_DIM, stoi)

#  Dataset & DataLoader
def encode_sentence(text, max_len, stoi):
    tokens = simple_tokenize(text.lower())
    ids = []
    for t in tokens[:max_len]:
        ids.append(stoi.get(t, stoi["<UNK>"]))
    if len(ids) < max_len:
        ids += [stoi["<PAD>"]] * (max_len - len(ids))
    return ids

class HumorDataset(Dataset):
    def __init__(self, texts, labels, max_len, stoi):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.stoi = stoi

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = encode_sentence(text, self.max_len, self.stoi)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )

train_dataset = HumorDataset(X_train, y_train, MAX_LEN, stoi)
dev_dataset   = HumorDataset(X_dev, y_dev, MAX_LEN, stoi)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

# BiLSTM Regression Model
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=1, bidirectional=True, dropout=0.3,
                 embedding_matrix=None, freeze_emb=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)           # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(emb)       # h_n: (num_layers*dirs, batch, hidden)
        if self.lstm.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden)
        else:
            h = h_n[-1, :, :]
        h = self.dropout(h)
        out = self.fc(h).squeeze(1)               # (batch,)
        return out

model = BiLSTMRegressor(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    bidirectional=BIDIRECTIONAL,
    dropout=DROPOUT,
    embedding_matrix=embedding_matrix,
    freeze_emb=True
).to(DEVICE)

print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# Training & Evaluation Loop
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = model(input_ids)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    return mae, rmse, all_preds, all_labels

best_dev_rmse = float("inf")
best_state_dict = None

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for input_ids, labels in train_loader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        preds = model(input_ids)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    dev_mae, dev_rmse, dev_preds, dev_labels = evaluate(model, dev_loader)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Dev MAE: {dev_mae:.4f} | Dev RMSE: {dev_rmse:.4f}"
    )

    if dev_rmse < best_dev_rmse:
        best_dev_rmse = dev_rmse
        best_state_dict = model.state_dict()

print("Best Dev RMSE:", best_dev_rmse)

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), "bilstm_glove_regressor.pt")
    print("Best BiLSTM+GloVe model saved to bilstm_glove_regressor.pt")
