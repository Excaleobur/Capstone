import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_dataset.csv' with the actgitual file)
df = pd.read_csv('Amir/Data/normalized_RB_data.csv')
# Select features (independent variables) and the target variable
X = df[['attempts', 'ypa','touchdowns','elusive_rating','fumbles']]  # Replace with actual features
y = df['Match']  # Replace with actual target variable

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

# Instantiate the model
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train.view(-1, 1))  # Calculate the binary cross-entropy loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    losses.append(loss.item())

# Make predictions on the test data
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred >= 0.5).float()  # Convert probabilities to binary predictions (0 or 1)

# Convert tensors back to NumPy arrays for evaluation
y_test = y_test.numpy()
y_pred = y_pred.numpy()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the loss curve (optional)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()