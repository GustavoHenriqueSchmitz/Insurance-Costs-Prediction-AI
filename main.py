from tensorflow import keras
import pandas as pd
import numpy as np
import pickle

# Load the model for use
loaded_model = keras.models.load_model('Insurance_cost_prediction_model')

# Load the train_columns
with open('train_columns.pkl', 'rb') as f:
    train_columns = pickle.load(f)

print("======================================================")
insurance_cost_values = []
for person_information in ["Age", "Sex", "BMI", "Children", "Smoker", "Region"]:
    insurance_cost_values.append(input(f"{person_information}: "))
# Convert your list into a DataFrame
insurance_cost_values = pd.DataFrame([insurance_cost_values], columns=["age", "sex", "bmi", "children", "smoker", "region"])
# Preprocess your data to match the training data
insurance_cost_values = pd.get_dummies(insurance_cost_values).reindex(columns=train_columns, fill_value=0)

# Make a prediction on a new data point
predictions = loaded_model.predict(np.array(insurance_cost_values).astype('float32'))
print(predictions)

