import streamlit as st
import joblib
import pandas as pd

# Load the trained model and feature columns
model = joblib.load('salary_prediction_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("Salary Prediction App")
st.write("Enter employee details to predict the salary.")

# Age slider with smooth transitions
age = st.slider('Age', min_value=20, max_value=65, value=30, 
               help="Minimum age is 20 as per professional standards")

# Experience slider (only appears if age > 20)
experience_enabled = age > 20
years_exp = 0

if experience_enabled:
    max_exp = max(0, age - 20)  # Assuming minimum working age is 20
    years_exp = st.slider('Years of Experience', 
                         min_value=0, 
                         max_value=max(40, max_exp),  # Cap at 40 for realism
                         value=min(5, max_exp),
                         help="Professional experience years")

# Education level conditional options
education_options = ["Bachelor's"]
if age >= 23:
    education_options += ["Master's", "PhD"]

education = st.selectbox('Education Level', education_options,
                        help="Advanced degrees require minimum age of 23")

# Job title conditional options
base_job_titles = ['Software Engineer', 'Data Analyst', 'Sales Associate']
senior_job_titles = ['Senior Manager', 'Director']

# Show senior roles only if:
# - Age > 25 
# - Has at least Master's degree
show_senior_roles = age > 25 and education in ["Master's", "PhD"]
job_titles = base_job_titles + senior_job_titles if show_senior_roles else base_job_titles

job_title = st.selectbox('Job Title', job_titles,
                        help="Senior roles require age > 25 and advanced degree")

# Gender selection
gender = st.selectbox('Gender', ['Male', 'Female'])

def preprocess_input(age, years_exp, gender, education, job_title):
    features = {col: 0 for col in feature_columns}
    
    # Set numerical features
    features['Age'] = age
    features['Years of Experience'] = years_exp if experience_enabled else 0
    
    # Set categorical features
    if 'Gender_Male' in features:
        features['Gender_Male'] = 1 if gender == 'Male' else 0
        
    if education == "Master's" and "Education Level_Master's" in features:
        features["Education Level_Master's"] = 1
    elif education == "PhD" and "Education Level_PhD" in features:
        features["Education Level_PhD"] = 1
        
    job_col = f"Job Title_{job_title}"
    if job_col in features:
        features[job_col] = 1

    return pd.DataFrame([features], columns=feature_columns)

if st.button('Predict Salary'):
    input_df = preprocess_input(age, years_exp, gender, education, job_title)
    
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Annual Salary: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
