from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json

app = Flask(__name__)
CORS(app)

model = joblib.load("./model.pkl")
scaler = joblib.load("./scaler.pkl")

aami_mapping = {
    0.0: 'N (Normal beat)',
    1.0: 'S (Supraventricular ectopic beat)',
    2.0: 'V (Ventricular ectopic beat)',
    3.0: 'F (Fusion beat)',
    4.0: 'Q (Unknown beat)'
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json.get("ecg", [])
        print(f"Received ECG data of length: {len(data)}")

        if len(data) != 187:
            return jsonify({"error": f"ECG must have exactly 187 values. You provided {len(data)}."}), 400

        ecg = np.array(data, dtype=float).reshape(1, -1)
        ecg_scaled = scaler.transform(ecg)

        pred = model.predict(ecg_scaled)[0]
        prob = model.predict_proba(ecg_scaled)[0]
        confidence = float(np.max(prob)) * 100

        # Determine urgency style based on class
        urgency = "success" if pred == 0.0 else "danger"

        return jsonify({
            "prediction": int(pred),
            "confidence": round(confidence, 2),
            "result": aami_mapping.get(pred, "Unknown"),
            "urgency": urgency
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/results", methods=["GET"])
def results():
    try:
        with open("../Model/results.json") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "Results not ready yet. Please run train.py first."}), 500

if __name__ == "__main__":
    app.run(debug=True)