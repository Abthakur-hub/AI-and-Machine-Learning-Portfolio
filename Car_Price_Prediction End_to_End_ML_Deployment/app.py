from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)


model = pickle.load(
    open("model/car_price_model.pkl","rb")
)



@app.route("/")
def home():

    return render_template("index.html")




@app.route("/predict", methods=["POST"])

def predict():


    year = int(request.form["year"])

    km = int(request.form["km"])

    fuel = int(request.form["fuel"])

    seller = int(request.form["seller"])

    transmission = int(request.form["transmission"])

    owner = int(request.form["owner"])

    engine = int(request.form["engine"])



    input_data = np.array(
        [[
        year,
        km,
        fuel,
        seller,
        transmission,
        owner,
        engine
        ]]
    )



    prediction = model.predict(input_data)


    price = round(prediction[0])



    return render_template(
        "index.html",
        prediction=f"Estimated Price ₹ {price}"
    )





if __name__=="__main__":

    app.run(
        host="0.0.0.0",
        port=10000
    )