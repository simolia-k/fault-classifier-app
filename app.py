from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# 훈련된 모델, 스케일러, 라벨 인코더 로드
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Model and preprocessors loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Make sure model.pkl, scaler.pkl, and label_encoder.pkl are in the same directory.")
    model = None

# 데이터 컬럼명 (모델 훈련시 사용된 순서와 정확히 일치해야 함)
feature_names = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']

@app.route('/')
def home():
    if model is None:
        return "<h1>Error: Model not loaded</h1><p>Please check the server logs. Make sure model files (.pkl) are present.</p>"
    return render_template('index.html', columns=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[col]) for col in feature_names]
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction_encoded = model.predict(scaled_features)
        prediction = label_encoder.inverse_transform(prediction_encoded)
        return render_template('index.html', columns=feature_names, prediction_text=f'✅ Predicted Fault Type: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', columns=feature_names, prediction_text=f'❌ Error: {e}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)