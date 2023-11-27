import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Capstone/JeevSauce/G/GData/normalized_G_data.csv')

# Select features and target variable
X = df[['player_game_count', 'block_percent', 'declined_penalties', 'hits_allowed', 'hurries_allowed', 'non_spike_pass_block', 'penalties', 'pressures_allowed', 'sacks_allowed', 'snap_counts_block', 'snap_counts_ce', 'snap_counts_lt', 'snap_counts_pass_block', 'snap_counts_pass_play', 'snap_counts_run_block']]
y = df['Match']

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                          dropout=dropout_prob, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Multiply by 2 for bidirectional
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Tensor conversion and reshaping
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Instantiate the model
    model = RNNModel(input_dim=X_train.shape[2], hidden_dim=128, num_layers=3, dropout_prob=0.4)

    # Adjust class weights
    total_samples = len(y_train)
    neg_samples = float((y_train == 0).sum())
    pos_samples = float((y_train == 1).sum())
    pos_weight = (total_samples / pos_samples) * (neg_samples / total_samples)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training loop with early stopping
    num_epochs = 100
    best_val_loss = float('inf')
    patience, trials = 5, 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()

        # Early stopping based on validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test.view(-1, 1))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trials = 0
            else:
                trials += 1
                if trials >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_test))
        threshold = 0.45  # Adjusted classification threshold
        y_pred_class = (y_pred >= threshold).float()
        y_test_np = y_test.numpy()
        y_pred_np = y_pred_class.numpy()

        # Metrics
        cm = confusion_matrix(y_test_np, y_pred_np)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test_np, y_pred_np, average='binary')
        results.append((cm, precision, recall, fscore))

# Average results from cross-validation
avg_cm = sum([r[0] for r in results]) / len(results)
avg_precision = sum([r[1] for r in results]) / len(results)
avg_recall = sum([r[2] for r in results]) / len(results)
avg_fscore = sum([r[3] for r in results]) / len(results)

print("Average Confusion Matrix:\n", avg_cm)
print(f"Average Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_fscore:.4f}")
