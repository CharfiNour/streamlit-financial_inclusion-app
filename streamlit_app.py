import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get label encodings
df = pd.read_csv("Financial_inclusion_dataset.csv")

# Manually re-create the label encodings as used during training
def get_label_dict(column):
    unique_vals = sorted(df[column].dropna().unique())
    return {val: i for i, val in enumerate(unique_vals)}

# Label mappings
country_dict = get_label_dict("country")
cellphone_dict = get_label_dict("cellphone_access")
gender_dict = get_label_dict("gender_of_respondent")
location_dict = get_label_dict("location_type")
marital_dict = get_label_dict("marital_status")
education_dict = get_label_dict("education_level")
job_dict = get_label_dict("job_type")

# Streamlit UI
st.title("📊 Financial Inclusion Prediction")
st.write("Will this person have a bank account?")

# Input widgets
country = st.selectbox("Country", list(country_dict.keys()))
cellphone = st.selectbox("Cellphone Access", list(cellphone_dict.keys()))
age = st.slider("Age of Respondent", 10, 100, 30)
household = st.slider("Household Size", 1, 20, 4)
marital_status = st.selectbox("Marital Status", list(marital_dict.keys()))
education = st.selectbox("Education Level", list(education_dict.keys()))
job = st.selectbox("Job Type", list(job_dict.keys()))
location = st.selectbox("Location Type", list(location_dict.keys()))
gender = st.selectbox("Gender", list(gender_dict.keys()))

# Encode input
input_data = np.array([[
    country_dict[country],
    cellphone_dict[cellphone],
    age,
    household,
    marital_dict[marital_status],
    education_dict[education],
    job_dict[job],
    location_dict[location],
    gender_dict[gender]
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    label = "✅ Has Bank Account" if prediction[0] == 1 else "❌ No Bank Account"
    st.success(f"Prediction: {label}")
