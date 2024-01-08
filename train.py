# load libraries
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

# load dataset
df = pd.read_csv("insurance.csv")

# Create X & y
features = df.drop("charges", axis=1)
labels = df["charges"]

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# Define preprocessing for numeric columns
numeric_features = ['age', 'bmi', 'children']
numeric_transformer = MinMaxScaler()

# Define preprocessing for categorical columns
categorical_features = ['sex', 'smoker', 'region']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine all preprocessing into one transformer
preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features)
)

# Preprocess the data
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(11, activation='relu', input_shape=[x_train.shape[1]]),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss='mean_squared_error',  # Loss function
    optimizer='adam',  # Optimizer
    metrics=['mean_absolute_error']  # Metric to monitor
)

# Train the model
history = model.fit(x_train, y_train, epochs=300, validation_split = 0.2)

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

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
            model.save('Insurance_cost_prediction_model')
            print("Model saved.")
            break
        else:
            print("Discarding Model...")
            break
