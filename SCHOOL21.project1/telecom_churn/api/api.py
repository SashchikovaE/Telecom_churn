from flask import Flask, render_template, request
import joblib
import pandas as pd
import sys
import os
from pathlib import Path
#current_dir = os.path.dirname(os.path.abspath(__file__))
#src_path = os.path.join(current_dir, 'src')
#sys.path.append(src_path)

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
src_path = parent_dir / 'src'
sys.path.append(str(src_path))
from preprocessing import DataPreprocessor

app = Flask(__name__)

def load():
    cur_file = Path(__file__)
    model_dir = cur_file.parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'log_regression_original.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    if model_path.stat().st_size == 0:
        raise ValueError(f"Файл модели пуст: {model_path}")
    model = joblib.load(model_path)
    return model

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    cols = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': int(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }
    model = load()
    cols_df = pd.DataFrame([cols])
    data = DataPreprocessor(df=cols_df)
    BINARY_COLUMNS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    ONE_HOT_COLUMNS = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for i in data.df.columns.tolist():
        if i in ONE_HOT_COLUMNS:
            data.encode_categorical_data(
                i, binary_mapping, is_binary_category=0, drop_first=False)
        if i in BINARY_COLUMNS:
            data.encode_categorical_data(
                i,
                binary_mapping,
                is_binary_category=1, drop_first=False)
    data.convert_object_to_float()
    data.add_new_features()
    data.add_new_features(is_tenure_status=0)
    missing_cols = set(model.feature_names_in_) - set(data.df.columns)
    for cols in missing_cols:
        data.df[cols] = 0
    data.df = data.df[model.feature_names_in_]
    data.normalize_data(loaded=True)
    prediction = model.predict(data.df)[0]
    probability = model.predict_proba(data.df)[0, 1]
    result = {
        'prediction': 'Клиент уйдет' if prediction == 1 else 'Клиент останется',
        'probability': round(probability * 100, 2),
        'confidence': 'высокая' if probability > 0.7 else 'средняя' if probability > 0.5 else 'низкая'
    }
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
