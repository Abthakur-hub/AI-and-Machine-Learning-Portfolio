import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

import pickle


# Load dataset

data = pd.read_csv("dataset/car_data.csv")


# Convert text into numbers

encoder = LabelEncoder()


for col in data.select_dtypes(include="object"):
    data[col] = encoder.fit_transform(data[col])


# Split data

X = data.drop("price",axis=1)

y = data["price"]


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train model

model = LinearRegression()

model.fit(
    X_train,
    y_train
)



# Save model

pickle.dump(
    model,
    open("model/car_price_model.pkl","wb")
)


print("Model saved successfully")