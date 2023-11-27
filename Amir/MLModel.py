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
import xgboost as xgb
# Load the dataset
df = pd.read_csv('Amir/Data/normalized_RB_data.csv')

# Select features and the target variable
X = df[['player_game_count', 'attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent', 'breakaway_yards', 'designed_yards', 'drops', 'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'elusive_rating', 'explosive', 'first_downs', 'fumbles', 'gap_attempts', 'grades_hands_fumble', 'grades_offense', 'grades_offense_penalty', 'grades_pass', 'grades_pass_block', 'grades_pass_route', 'grades_run', 'grades_run_block', 'longest', 'penalties', 'rec_yards', 'receptions', 'routes', 'run_plays', 'scramble_yards', 'scrambles', 'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact', 'yco_attempt', 'ypa', 'yprr', 'zone_attempts']]  # Replace with actual features
y = df['Match']  # Replace with actual target variable

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
num_epochs = 100
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
ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
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



from sklearn.svm import SVC
import xgboost as xgb

# Other imports and your existing code remains the same

# XGBoost Model
xgb_model = xgb.XGBClassifier(scale_pos_weight=sum(y_train == 0) / sum(y_train == 1))
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

# Evaluate XGBoost Model
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_f1 = f1_score(y_test, xgb_y_pred)
xgb_conf_matrix = confusion_matrix(y_test, xgb_y_pred)

print("\nXGBoost Model Evaluation:")
print(f"Accuracy: {xgb_accuracy * 100:.2f}%")
print(f"F1 Score: {xgb_f1:.2f}")
print("Confusion Matrix:")
print(xgb_conf_matrix)

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
print(f"F1 Score: {svm_f1:.2f}")
print("Confusion Matrix:")
print(svm_conf_matrix)

# Plot confusion matrix for XGBoost model
plot_confusion_matrix(y_test, xgb_y_pred, classes=[0, 1], model_name='XGBoost')

# Plot confusion matrix for SVM model
plot_confusion_matrix(y_test, svm_y_pred, classes=[0, 1], model_name='SVM')
# Example: Plot confusion matrix for Random Forest model
plot_confusion_matrix(y_test, rf_y_pred, classes=[0, 1], model_name='Random Forest')

# Example: Plot confusion matrix for AdaBoost model
plot_confusion_matrix(y_test, ab_y_pred, classes=[0, 1], model_name='AdaBoost')