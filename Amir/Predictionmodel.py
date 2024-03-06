import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset (replace 'your_dataset.csv' with the actgitual file)
df = pd.read_csv('Amir/Data/normalized_RB_data.csv')
# Select features (independent variables) and the target variable
X = df[['player_game_count', 'attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent', 'breakaway_yards', 'designed_yards', 'drops', 'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'elusive_rating', 'explosive', 'first_downs', 'fumbles', 'gap_attempts', 'grades_hands_fumble', 'grades_offense', 'grades_offense_penalty', 'grades_pass', 'grades_pass_block', 'grades_pass_route', 'grades_run', 'grades_run_block', 'longest', 'penalties', 'rec_yards', 'receptions', 'routes', 'run_plays', 'scramble_yards', 'scrambles', 'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact', 'yco_attempt', 'ypa', 'yprr', 'zone_attempts']] # Replace with actual features
y = df['Match']  # Replace with actual target variable


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


column_names = df.columns.tolist()
print(column_names)

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

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the loss curve (optional)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
# Save the model's state dictionary
torch.save(model.state_dict(), '/Users/jeeva/NextFootball/NewFlaskProject/PredictionModel.pth')
