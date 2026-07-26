# 🚗 Car Price Prediction System

A Machine Learning based web application that predicts the selling price of a car using user input features.

## 📌 Project Overview

This project uses Supervised Learning and Flask to create an end-to-end ML web application.

Flow:

Dataset → Model Training → Pickle Model → Flask Backend → HTML/CSS UI → Prediction


## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Flask
- HTML5
- CSS3
- GitHub
- Render


## 🤖 Machine Learning Model

Algorithm Used:

- Linear Regression


Model Deployment:

- Pickle (.pkl)


Input Features:

- Car Year
- KM Driven
- Fuel Type
- Seller Type
- Transmission
- Owner
- Engine


Output:

- Predicted Car Price


## 📂 Project Structure
Car_Price_Prediction/

│
├── dataset/
│ └── car_data.csv
│
├── model/
│ └── car_price_model.pkl
│
├── static/
│ └── style.css
│
├── templates/
│ └── index.html
│
├── app.py
├── train_model.py
├── requirements.txt
└── Procfile



## ▶️ Run Locally

Clone repository:

```bash
git clone https://github.com/Abthakur-hub/Car_Price_Prediction.git

Install dependencies:

pip install -r requirements.txt

Run Flask:

python3 app.py

Open browser:

http://127.0.0.1:8000
🌐 Deployment
