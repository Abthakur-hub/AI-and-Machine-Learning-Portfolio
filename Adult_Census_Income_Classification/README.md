# Adult Census Income Classification

A Machine Learning project that predicts whether an individual's annual income is **greater than \$50K** or **less than or equal to \$50K** using demographic and employment-related information.

---

## 👨‍💻 Author

**Abhishek Thakur**
**IN26011189**

GitHub: https://github.com/Abthakur-hub

---

## 📌 Project Overview

The goal of this project is to build a classification model using the **Adult Census Income Dataset**. The model learns from various features such as age, education, occupation, work hours, and marital status to predict a person's income category.

---

## 📂 Dataset

- **Dataset:** Adult Census Income Dataset
- **Source:** UCI Machine Learning Repository
- **Records:** 32,561
- **Features:** 14 Input Features + 1 Target Feature

### Target Variable

- `<=50K`
- `>50K`

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Joblib

---

## 📁 Project Structure

```
Adult_Census_Income_Classification/
│
├── data/
│   └── adult.data
│
├── notebook/
│   └── Adult_Census_Income_Classification.ipynb
│
├── models/
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Machine Learning Workflow

1. Import Libraries
2. Load Dataset
3. Data Cleaning
4. Handle Missing Values
5. Exploratory Data Analysis (EDA)
6. Label Encoding
7. Feature Scaling
8. Train-Test Split
9. Train Multiple Machine Learning Models
10. Compare Model Performance
11. Save the Best Model
12. Deploy Using Flask

---

## 🤖 Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📊 Model Evaluation

The models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

Random Forest achieved the best performance and was selected as the final model.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Adult_Census_Income_Classification.git
```

Move into the project directory

```bash
cd Adult_Census_Income_Classification
```

Create a virtual environment

```bash
python3 -m venv venv
```

Activate the virtual environment

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the model

```bash
cd src
python3 train.py
```

Run the Flask application

```bash
cd ..
python3 app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📈 Features

- Data Cleaning
- Missing Value Handling
- Label Encoding
- Feature Scaling
- Multiple Machine Learning Models
- Model Comparison
- Model Persistence using Joblib
- Flask Web Application

---

## 📌 Future Improvements

- Improve the user interface
- Use dropdown menus for categorical inputs
- Deploy the application on Render
- Add more machine learning models
- Perform hyperparameter tuning

---


## ⭐ If you found this project useful, consider giving it a star!