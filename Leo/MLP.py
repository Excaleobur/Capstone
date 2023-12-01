import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
import numpy as np

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
    remainder='passthrough'  # The rest of the columns are passed through
)

# Transform the dataset with the column_transformer
features_encoded = column_transformer.fit_transform(features)

# Separate the target variable
y = ncaa['nfl']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, y, test_size=0.2, random_state=42)

# now to build the model architecture
import tensorflow as tf

# Define the number of inputs, hidden layer neurons, and outputs
# input dimenions is the number of columns in the dataset
input_dimension = X_train.shape[1]
hidden_neurons = 128
num_classes = 2

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_neurons, activation='relu', input_shape=(input_dimension,)),
    # Applied to the previous layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(hidden_neurons, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model using a standard optimizer and loss function for classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Compute class weights
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weights[i] for i in range(len(weights))}

# Now pass these computed class weights to the model.fit() function
history = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2, class_weight=class_weights)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])

# Plot the accuracy and loss curves
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# confusion matrix and f1 score
from sklearn.metrics import confusion_matrix, f1_score

# When you want to actually predict stuff with no labels
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))





