from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Categorical → Numeric mappings
SEX_MAP    = {'M': 1, 'F': 0}
CP_MAP     = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
ECG_MAP    = {'Normal': 0, 'ST': 1, 'LVH': 2}
ANGINA_MAP = {'Y': 1, 'N': 0}
SLOPE_MAP  = {'Up': 2, 'Flat': 1, 'Down': 0}

FEATURE_COLUMNS = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# Load model
with open('Heart_Project.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Extract form inputs
            age         = float(request.form['Age'])
            sex_raw     = request.form['Sex']
            cp_raw      = request.form['ChestPainType']
            resting_bp  = float(request.form['RestingBP'])
            chol        = float(request.form['Cholesterol'])
            fasting_bs  = float(request.form['FastingBS'])
            ecg_raw     = request.form['RestingECG']
            max_hr      = float(request.form['MaxHR'])
            angina_raw  = request.form['ExerciseAngina']
            oldpeak     = float(request.form['Oldpeak'])
            slope_raw   = request.form['ST_Slope']  # ✅ Fixed typo here

            # Map categoricals
            sex     = SEX_MAP.get(sex_raw)
            cp      = CP_MAP.get(cp_raw)
            ecg     = ECG_MAP.get(ecg_raw)
            angina  = ANGINA_MAP.get(angina_raw)
            slope   = SLOPE_MAP.get(slope_raw)

            # Build DataFrame
            feature_values = [
                age, sex, cp, resting_bp, chol,
                fasting_bs, ecg, max_hr, angina,
                oldpeak, slope
            ]
            X_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

            # Predict
            pred_prob  = model.predict_proba(X_df)[0, 1]
            pred_class = model.predict(X_df)[0]

            prediction = {
                'class': int(pred_class),
                'probability': round(pred_prob, 3)
            }

        except Exception as e:
            prediction = {'error': str(e)}

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)