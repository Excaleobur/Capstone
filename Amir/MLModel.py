import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


# Load the dataset
df = pd.read_csv('Amir/Data/combine_added.csv')
print(df.columns)

# Select features and the target variable
#X = df[['attempts', 'yards', 'touchdowns', 'ypa', 'breakaway_yards', 'avoided_tackles', 'elu_rush_mtf', 'yards_after_contact', 'receptions', 'rec_yards', 'targets']]
X = df[['attempts', 'yards', 'touchdowns', 'ypa', 'breakaway_yards', 'avoided_tackles', 
'elu_rush_mtf', 'yards_after_contact', 'receptions', 'rec_yards', 'targets','Height', 'Weight',
       'Hand Size', 'Arm Length', 'Wonderlic', '40Yard', 'Bench Press',
       'Vert Leap', 'Broad Jump', 'Shuttle', '3Cone', '60Yd Shuttle']]

#X = df[['player_game_count', 'attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent', 'breakaway_yards', 'designed_yards', 'drops', 'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'elusive_rating', 'explosive', 'first_downs', 'fumbles', 'gap_attempts', 'grades_hands_fumble', 'grades_offense', 'grades_offense_penalty', 'grades_pass', 'grades_pass_block', 'grades_pass_route', 'grades_run', 'grades_run_block', 'longest', 'penalties', 'rec_yards', 'receptions', 'routes', 'run_plays', 'scramble_yards', 'scrambles', 'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact', 'yco_attempt', 'ypa', 'yprr', 'zone_attempts']]  # Replace with actual features
y = df['Match']  

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)  # Training with NumPy arrays
rf_y_pred = rf_model.predict(X_test)

# Evaluate Random Forest Model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)

print("Random Forest Model Evaluation:")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
print(f"F1 Score: {rf_f1:.2f}")
print("Confusion Matrix:")
print(rf_conf_matrix)

class EnhancedNNModel(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedNNModel, self).__init__()
        # Increasing the depth and complexity of the network
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.batchnorm2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


input_dim = X_train_tensor.shape[1]
dl_model = EnhancedNNModel(input_dim)

# Define loss function and optimizer for deep learning model
criterion = nn.BCELoss()
optimizer = optim.SGD(dl_model.parameters(), lr=0.01)

# Training loop for deep learning model
num_epochs = 500
dl_losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = dl_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    loss.backward()
    optimizer.step()
    dl_losses.append(loss.item())

# Predictions for deep learning model
with torch.no_grad():
    dl_y_pred = dl_model(X_test_tensor)
    dl_y_pred = (dl_y_pred >= 0.5).float()

# Convert tensors back to NumPy arrays for evaluation
dl_y_pred = dl_y_pred.numpy()
dl_y_test = y_test_tensor.numpy()

# Evaluate Deep Learning Model
dl_accuracy = accuracy_score(dl_y_test, dl_y_pred)
dl_f1 = f1_score(dl_y_test, dl_y_pred)
dl_conf_matrix = confusion_matrix(dl_y_test, dl_y_pred)

print("\nDeep Learning Model Evaluation:")
print(f"Accuracy: {dl_accuracy * 100:.2f}%")
print(f"F1 Score: {dl_f1:.2f}")
print("Confusion Matrix:")
print(dl_conf_matrix)

# Visualize the loss curve for deep learning model
plt.plot(dl_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss for Deep Learning Model")
plt.show()


# AdaBoost Model
ab_model = AdaBoostClassifier(n_estimators=500, random_state=42)
ab_model.fit(X_train, y_train)  # Training
ab_y_pred = ab_model.predict(X_test)

# Evaluate AdaBoost Model
ab_accuracy = accuracy_score(y_test, ab_y_pred)
ab_f1 = f1_score(y_test, ab_y_pred)
ab_conf_matrix = confusion_matrix(y_test, ab_y_pred)

print("AdaBoost Model Evaluation:")
print(f"Accuracy: {ab_accuracy * 100:.2f}%")
print(f"F1 Score: {ab_f1:.2f}")
print("Confusion Matrix:")
print(ab_conf_matrix)
# Save the models
#joblib.dump(rf_model, '/Users/amirrezarafati/Downloads/CapsotneModel/RB/Repo/Capstone/Amir/Data/RandomForestModel.pkl')
#torch.save(dl_model.state_dict(), '/Users/amirrezarafati/Downloads/CapsotneModel/RB/Repo/Capstone/Amir/Data/DeepLearningModel.pth')



def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Other imports and your existing code remains the same


# Support Vector Machine (SVM) Model
svm_model = SVC(class_weight='balanced')
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

# Evaluate SVM Model
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_f1 = f1_score(y_test, svm_y_pred)
svm_conf_matrix = confusion_matrix(y_test, svm_y_pred)

print("\nSVM Model Evaluation:")
print(f"Accuracy: {svm_accuracy * 100:.2f}%")
print(f"F1 Score: ", svm_f1)
print("Confusion Matrix:", svm_conf_matrix)


joblib.dump(svm_model, '/Users/amirrezarafati/Downloads/CapsotneModel/RB/Repo/Capstone/Amir/Data/PredictionModelSVM.pkl')

# Plot confusion matrix for SVM model
plot_confusion_matrix(y_test, svm_y_pred, classes=[0, 1], model_name='SVM')
# Example: Plot confusion matrix for Random Forest model
plot_confusion_matrix(y_test, rf_y_pred, classes=[0, 1], model_name='Random Forest')

# Example: Plot confusion matrix for AdaBoost model
plot_confusion_matrix(y_test, ab_y_pred, classes=[0, 1], model_name='AdaBoost')


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



plot_actual_vs_predicted_distribution_plotly(y_test, svm_y_pred, 'SVM')
