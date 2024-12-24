from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define labels for input fields
labels = [
    "Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)", "Resting Blood Pressure",
    "Serum Cholestoral in mg/dl", "Fasting Blood Sugar (1=True, 0=False)", "Resting Electrocardiographic Results (0-2)",
    "Maximum Heart Rate Achieved", "Exercise Induced Angina (1=True, 0=False)", "Oldpeak (ST depression)",
    "Slope of Peak Exercise ST Segment", "Number of Major Vessels Colored by Fluoroscopy", "Thalassemia (0-3)"
]

@app.route('/')
def index():
    return render_template('index.html', labels=labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = [float(request.form[label]) for label in labels]
        
        # Rescale the input data
        input_data_scaled = scaler.transform([input_data])
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        
        result = "The patient has heart disease." if prediction == 1 else "The patient does not have heart disease."
        
        return jsonify({"result": result})
    
    except ValueError:
        return jsonify({"error": "Invalid input. Please enter valid numeric values for all fields."})

if __name__ == '__main__':
    app.run(debug=True)
