from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import json

app = Flask(__name__)
model = load_model("model/weather_forecast_model.keras")

# Load precomputed metrics once
with open("metrics.json", "r") as f:
    cached_metrics = json.load(f)

@app.route("/")
def home():
    return "✅ Weather Forecasting Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        input_seq = np.array(data['input']).reshape(1, 120, 7)
        prediction = model.predict(input_seq)
        return jsonify({'predicted_temperature': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(cached_metrics)

if __name__ == "__main__":
    app.run(debug=True)
