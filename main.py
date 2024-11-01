from tensorflow import keras
import pandas as pd
import pickle

# Load the model for use
loaded_model = keras.models.load_model('Insurance_cost_prediction_model.h5')

# Load the preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

print("======================================================")
insurance_cost_values = []
for person_information in ["Age", "Sex", "BMI", "Children", "Smoker", "Region"]:
    insurance_cost_values.append(input(f"{person_information}: "))
# Convert your list into a DataFrame
insurance_cost_values = pd.DataFrame([insurance_cost_values], columns=["age", "sex", "bmi", "children", "smoker", "region"])

# Preprocess your data to match the training data
insurance_cost_values = preprocessor.transform(insurance_cost_values)

# Make a prediction on a new data point
predictions = loaded_model.predict(insurance_cost_values)
print("The predicted insurance cost is: $", predictions[0][0])
