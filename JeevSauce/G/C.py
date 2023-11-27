import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

# Load the dataset
df = pd.read_csv('Capstone/JeevSauce/G/GData/normalized_G_data.csv')

# Select features and target variable
X = df[['player_game_count', 'block_percent', 'declined_penalties', 'hits_allowed', 'hurries_allowed', 'non_spike_pass_block', 'penalties', 'pressures_allowed', 'sacks_allowed', 'snap_counts_block', 'snap_counts_ce', 'snap_counts_lt', 'snap_counts_pass_block', 'snap_counts_pass_play', 'snap_counts_run_block']]
y = df['Match']

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Reshape X_train and X_test to add a sequence dimension
X_train = X_train[:, np.newaxis, :]  # Shape: (batch_size, 1, num_features)
X_test = X_test[:, np.newaxis, :]    # Shape: (batch_size, 1, num_features)

# Define the complex model architecture
class ComplexModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob):
        super(ComplexModel, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1)  # Reduce kernel size to 1
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)  # Reduce kernel size to 1
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # Multiply by 2 for bidirectional LSTM
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for Conv1D
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = x.permute(0, 2, 1)  # Reshape back for LSTM
        h0 = torch.zeros(2 * num_layers, x.size(0), hidden_dim).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(2 * num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out



# Tensor conversion and reshaping
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the complex model
input_dim = X_train.shape[2]  # Number of features
hidden_dim = 128
num_layers = 2
dropout_prob = 0.4
model = ComplexModel(input_dim, hidden_dim, num_layers, dropout_prob)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    threshold = 0.5  # Adjust classification threshold if needed
    y_pred = (y_pred >= threshold).float()

    # Metrics
    cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test.numpy(), y_pred.numpy(), average='binary')

print("Complex Model Confusion Matrix:\n", cm)
print(f"Complex Model Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fscore:.4f}")
