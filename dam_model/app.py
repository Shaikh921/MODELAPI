from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("dam_overflow_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

FEATURE_ORDER = [
    "rain_mm_24h","rain_intensity","humidity","wind_speed","air_temp_c",
    "inflow_m3s","outflow_m3s","water_level_m","upstream_flow_index"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message":"Dam Overflow Predictor API. POST /predict with JSON features."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Validate features
    for f in FEATURE_ORDER:
        if f not in data:
            return jsonify({"error": f"Missing feature: {f}"}), 400
    X = np.array([[ float(data[f]) for f in FEATURE_ORDER ]])
    Xs = scaler.transform(X)
    proba = float(model.predict_proba(Xs)[:,1])
    label = int(proba >= 0.5)
    return jsonify({"probability": proba, "overflow_label": label})

if __name__ == "__main__":
    app.run(debug=True)
