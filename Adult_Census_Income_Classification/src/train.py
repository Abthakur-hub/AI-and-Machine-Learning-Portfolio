import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# Load Dataset
# ==============================

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

df = pd.read_csv(
    "../data/adult.data",
    names=columns,
    header=None,
    skipinitialspace=True
)

# ==============================
# Data Cleaning
# ==============================

df.replace("?", np.nan, inplace=True)

for column in df.select_dtypes(include="object").columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# ==============================
# Label Encoding
# ==============================

encoders = {}

for column in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# ==============================
# Features and Target
# ==============================

X = df.drop("income", axis=1)
y = df["income"]

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# Train Model
# ==============================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==============================
# Evaluate
# ==============================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy : {accuracy*100:.2f}%")

# ==============================
# Save Model
# ==============================

joblib.dump(model, "../models/random_forest.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(encoders, "../models/encoders.pkl")

print("Model Saved Successfully!")