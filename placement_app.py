
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.title("Student Placement Prediction")
st.write("Decision Tree Classification")

df = pd.read_csv(Path(__file__).with_name("converted_data.csv"))
feature_columns = [
    "CGPA",
    "Attendance",
    "Internship",
    "Coding",
    "Aptitude",
    "Communication",
]
X = df[feature_columns]
y = df["Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

st.subheader("Enter New Student Details")
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
attendance = st.number_input(
    "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0
)
internship = st.selectbox("Internship", ["No", "Yes"])
coding = st.number_input(
    "Coding Skill Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0
)
aptitude = st.number_input(
    "Aptitude Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0
)
communication = st.number_input(
    "Communication Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0
)

if st.button("Predict Placement"):
    new_student = pd.DataFrame(
        [[cgpa, attendance, int(internship == "Yes"), coding, aptitude, communication]],
        columns=feature_columns,
    )
    prediction = model.predict(new_student)[0]
    if prediction == 1:
        st.success("Prediction: PLACED")
    else:
        st.error("Prediction: NOT PLACED")
