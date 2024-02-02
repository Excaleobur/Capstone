import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

# Read the dataset
ncaa = pd.read_csv('finalnormalizedncaa.csv')
ncaa = ncaa.fillna(0)

# Convert the specified categorical columns to string type
for col in ['position', 'team_name']:
    ncaa[col] = ncaa[col].astype(str)

# Identify categorical columns for one-hot encoding
categorical_columns = ['position', 'team_name']

# Drop the target column 'nfl' before applying transformations
features = ncaa.drop(['nfl', 'player'], axis=1)

# Apply one-hot encoding to categorical columns
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

# Transform the dataset with the column_transformer
features_encoded = column_transformer.fit_transform(features)
features_encoded = features_encoded.toarray()

# Separate the target variable
y = ncaa['nfl']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
# One way to solve this problem is by increasing the number of samples in the minority class (the class with fewer samples) 
# so that it is on par with the majority class. This process is known as over-sampling
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Calculate class weights manually
# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Create a dictionary with class weights
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Create RandomForestClassifier with class weights
rf = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)

# Fit the model
rf.fit(X_train_smote, y_train_smote)

importances = rf.feature_importances_
print("Feature importances:", importances)

# Neural Network Model Setup
input_dimension = X_train_smote.shape[1]
hidden_neurons = 128
num_classes = 2

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_neurons, activation='relu', input_shape=(input_dimension,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(hidden_neurons, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# Train the model with the SMOTE data
# Define class weights for the neural network
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_smote),
    y=y_train_smote
)
class_weights_dict = dict(enumerate(class_weights))

# Train the model with class weights
history = model.fit(X_train_smote, y_train_smote, epochs=100, batch_size=50, validation_split=0.2, class_weight=class_weights_dict)


# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])

# Plot the accuracy and loss curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Confusion matrix and F1 score
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
print(cm)
f1 = f1_score(y_test, y_pred_classes)
print(f1)

# Save the model
model.save('improved_model.h5')
