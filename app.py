import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load Dataset
data = pd.read_csv("student_data.csv")

# Features and Target
X = data[['StudyHours', 'Attendance', 'PreviousScore']]
y = data['Result']

# Train Model
model = DecisionTreeClassifier()
model.fit(X, y)

st.title("🎓 Student Performance Prediction System")

st.write("Enter student details:")

study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0)

if st.button("Predict Result"):
    prediction = model.predict(
        [[study_hours, attendance, previous_score]]
    )[0]

    st.success(f"Predicted Result: {prediction}")
    
