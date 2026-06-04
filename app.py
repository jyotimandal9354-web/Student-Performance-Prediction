from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load Dataset
data = pd.read_csv("student_data.csv")

# Features and Target
X = data[['StudyHours', 'Attendance', 'PreviousScore']]
y = data['Result']

# Train Model
model = DecisionTreeClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return "Student Performance Prediction System"

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.json['StudyHours'])
    attendance = float(request.json['Attendance'])
    previous_score = float(request.json['PreviousScore'])

    prediction = model.predict(
        [[study_hours, attendance, previous_score]]
    )[0]

    return jsonify({"Prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)