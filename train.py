import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle

# Getting the data
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# Preprocessing the data for train
x_train = pd.get_dummies(train_data.drop('charges', axis=1))
train_columns = x_train.columns  # Save the column names
x_train = np.array(x_train).astype('float32')  # Convert to numpy array
y_train = train_data['charges']

# Save the train_columns for later use
with open('train_columns.pkl', 'wb') as f:
    pickle.dump(train_columns, f)

# Preprocessing the data for test
x_test = np.array(pd.get_dummies(test_data.drop('charges', axis=1))).astype('float32')
y_test = test_data["charges"]

# Create the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(x_train[0])]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss='mean_squared_error',  # Loss function
    optimizer='adam',  # Optimizer
    metrics=['mean_absolute_error']  # Metric to monitor
)

# Train the model
history = model.fit(x_train, y_train, epochs=1000, validation_split = 0.2)

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=2)

# Print the test results
print("================= Test Results ==================")
print(f"Loss: {test_loss} | MAR: {test_mae}")
print("=================================================")

while True:
    answer = str(input("According the results, do you want to continue and save the model for use [y/n]: ")).lower()
    if answer != "n" and answer != "y":
        print("Digit a valid answer please.")
        continue
    else:
        if answer == "y":
            print("Saving Model...")
            # Save the model
            model.save('Insurance_cost_prediction_model')
            print("Model saved.")
            break
        else:
            print("Discarding Model...")
            break
