import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Load data
df = pd.read_csv("C:/Users/DELL/Desktop/data.csv/data.csv")
# Features and target
X = df[["HoursStudied"]]
y = df["ExamScore"]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Streamlit app
st.title("Exam Score Predictor")
st.write("Enter the number of hours studied to predict the exam score.")
hours = st.number_input(
    "Hours Studied:",
    min_value=0.0,
    step=0.1)
if st.button("Predict Score"):
    predicted_score = model.predict([[hours]])[0]
    st.success(f"Predicted Score: {predicted_score:.2f}")
# Display data
st.write("### Sample Data")
st.dataframe(df)
