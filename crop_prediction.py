# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the dataset
data = pd.read_csv(r'C:\Users\hrr75\Desktop\MCA PROJECTS\crop prediction\crop_data.csv') # Replace with your dataset file
print("Dataset loaded successfully!\n", data.head())

# Step 2: Preprocessing the Data
# Assume the dataset has columns: 'Rainfall', 'Temperature', 'pH', and 'Crop'
X = data[['Rainfall', 'Temperature', 'pH']]
y = pd.get_dummies(data['Crop'])  # One-hot encoding for categorical target

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Visualize Data (EDA)
plt.scatter(data['Rainfall'], data['Temperature'], c='blue')
plt.xlabel('Rainfall')
plt.ylabel('Temperature')
plt.title('Rainfall vs Temperature')
plt.show()

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Adjust for number of crops
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Step 7: Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Step 8: Save the Model
model.save('crop_prediction_model.h5')
print("Model saved as 'crop_prediction_model.h5'")
