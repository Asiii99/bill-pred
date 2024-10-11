import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model_path = 'model/electricity_model.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Please train the model first.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request!")  # Debugging line
    data = request.get_json(force=True)
    
    if 'SystemLoadEA' not in data:
        return jsonify({'error': "Please provide 'SystemLoadEA' for prediction."}), 400

    input_data = pd.DataFrame(data, index=[0])

    try:
        input_data['SystemLoadEA'] = pd.to_numeric(input_data['SystemLoadEA'], errors='coerce')
    except ValueError:
        return jsonify({'error': "Invalid input data format."}), 400
    
    prediction = model.predict(input_data[['SystemLoadEA']])
    
    return jsonify({'predicted_SMPEA': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
