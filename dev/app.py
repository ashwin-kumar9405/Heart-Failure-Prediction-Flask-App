from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('heart_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature order for input
features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

# Mapping for categorical features to numeric encoding
mapping = {
    'Sex': {'M': 1, 'F': 0},
    'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
    'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
    'ExerciseAngina': {'N': 0, 'Y': 1},
    'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = []
        for feature in features:
            val = request.form.get(feature)
            if feature in mapping:
                val = mapping[feature].get(val, 0)
            else:
                val = float(val)
            data.append(val)
        
        # Scale features
        data_scaled = scaler.transform([data])
        
        # Predict
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][1]
        
        result = 'High risk of heart failure' if prediction == 1 else 'Low risk of heart failure'
        return render_template('index.html', prediction_text=result, probability=round(probability*100, 2))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: ' + str(e))

if __name__ == '__main__':
    app.run(debug=True)