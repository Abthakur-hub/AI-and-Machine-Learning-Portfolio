import pandas as pd
import joblib

# =====================================
# Load Saved Files
# =====================================

model = joblib.load("../models/random_forest.pkl")
scaler = joblib.load("../models/scaler.pkl")
encoders = joblib.load("../models/encoders.pkl")

# =====================================
# Sample Input
# =====================================

sample = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# =====================================
# Convert to DataFrame
# =====================================

df = pd.DataFrame([sample])

# =====================================
# Encode Categorical Features
# =====================================

categorical_columns = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"
]

for column in categorical_columns:
    df[column] = encoders[column].transform(df[column])

# =====================================
# Scale Features
# =====================================

scaled_data = scaler.transform(df)

# =====================================
# Prediction
# =====================================

prediction = model.predict(scaled_data)

if prediction[0] == 1:
    print("Predicted Income: >50K")
else:
    print("Predicted Income: <=50K")