import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

# Define the Neural Network Model
class NNModel(nn.Module):
    def __init__(self, input_dim):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Tensor conversion and reshaping
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Instantiate the model
model = NNModel(input_dim=X_train.shape[1])

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
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

print("Neural Network Confusion Matrix:\n", cm)
print(f"Neural Network Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fscore:.4f}")
