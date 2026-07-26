from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    workclass = request.form["workclass"]
    fnlwgt = int(request.form["fnlwgt"])
    education = request.form["education"]
    education_num = int(request.form["education_num"])
    marital_status = request.form["marital_status"]
    occupation = request.form["occupation"]
    relationship = request.form["relationship"]
    race = request.form["race"]
    sex = request.form["sex"]
    capital_gain = int(request.form["capital_gain"])
    capital_loss = int(request.form["capital_loss"])
    hours_per_week = int(request.form["hours_per_week"])
    native_country = request.form["native_country"]

    data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country
    }])

    categorical = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]

    for col in categorical:
        data[col] = encoders[col].transform(data[col])

    data = scaler.transform(data)

    prediction = model.predict(data)

    result = ">50K" if prediction[0] == 1 else "<=50K"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)