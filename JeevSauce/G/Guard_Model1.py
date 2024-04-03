import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


# Load the dataset (replace 'your_dataset.csv' with the actgitual file)
df = pd.read_csv('/Users/amirrezarafati/Downloads/CapsotneModel/RB/Repo/Capstone/JeevSauce/G/GData/normalized_G_data.csv')
# Select features (independent variables) and the target variable
X = df[['player_game_count','block_percent','declined_penalties','hits_allowed','hurries_allowed','non_spike_pass_block','penalties','pressures_allowed', 'sacks_allowed','snap_counts_block','snap_counts_ce','snap_counts_lt','snap_counts_pass_block', 'snap_counts_pass_play','snap_counts_run_block']]  # Replace with actual features
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


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def plot_actual_vs_predicted_distribution_plotly(y_actual, y_pred, model_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Actual Distribution: {model_name}',
                                                        f'Predicted Distribution: {model_name}'))

    # Actual distribution
    fig.add_trace(
        go.Histogram(x=y_actual, name='Actual', marker_color='blue'),
        row=1, col=1
    )

    # Predicted distribution
    fig.add_trace(
        go.Histogram(x=y_pred, name='Predicted', marker_color='red'),
        row=1, col=2
    )

    # Update titles and labels
    fig.update_layout(title_text=f"Actual vs. Predicted Distribution: {model_name}",
                      xaxis_title="Outcome",
                      yaxis_title="Count",
                      xaxis2_title="Outcome",
                      yaxis2_title="Count",
                      bargap=0.2)

    fig.update_xaxes(tickvals=[0, 1], ticktext=['Didn\'t Make It', 'Made It'], row=1, col=1)
    fig.update_xaxes(tickvals=[0, 1], ticktext=['Didn\'t Make It', 'Made It'], row=1, col=2)

    fig.show()



plot_actual_vs_predicted_distribution_plotly(y_test, y_pred, 'SVM')
