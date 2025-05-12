# Salary Prediction App

A web application that predicts annual salaries based on user inputs, built with Python, scikit-learn, XGBoost, and Streamlit.

## Features

- Age, experience, education level, job title, and gender inputs.
- Real-time salary prediction using a tuned XGBoost regression model.
- Conditional UI elements to reflect realistic professional scenarios.

## Installation

```bash
git clone https://github.com/preethammmm/salary-predictor
cd salary-prediction-app
pip install -r requirements.txt
```

## Usage

```bash
streamlit run salary_app.py
```

- Adjust inputs via sliders and dropdowns.
- Click **Predict Salary** to view the estimated annual salary.

## Project Structure

- `salaryprediction.py`: Data preprocessing, model training, evaluation, and saving the best model.
- `salary_app.py`: Streamlit app code for user interaction and model inference.
- `Salary Data.csv`: Dataset used for training and evaluation.
- `feature_columns.pkl`, `salary_prediction_model.pkl`: Artifacts loaded by the web app.
