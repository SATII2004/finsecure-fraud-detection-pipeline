from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join('src', 'models', 'fraud_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully for API.")
except:
    print("Model not found! Run run_pipeline.py first.")

@app.route('/')
def home():
    return "FinSecure Fraud Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive transaction data and return fraud status.
    Expects a JSON with a list of 30 features (Time, V1-V28, Amount).
    """
    try:
        data = request.get_json()
        # Convert input to numpy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        result = {
            'prediction': 'FRAUD' if int(prediction[0]) == 1 else 'LEGIT',
            'fraud_probability': float(probability[0][1]),
            'status': 200
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 400})

if __name__ == '__main__':
    # Running on port 5000
    app.run(debug=True, port=5000)