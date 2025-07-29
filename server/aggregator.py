# aggregator.py

import numpy as np
import pickle
from typing import List
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Store global model (initially empty)
GLOBAL_MODEL_PATH = "models/anomaly_model.pkl"
AGGREGATED_DATA_PATH = "data/aggregated_logs.npy"

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Initialize memory for FL rounds
client_updates = []


def simple_aggregate(updates: List[np.ndarray]) -> np.ndarray:
    """
    Federated Averaging for anomaly detection models
    Each update is assumed to be a numpy array of anomaly scores or features.
    """
    if not updates:
        return None
    return np.mean(np.array(updates), axis=0)


@app.route("/submit_update", methods=["POST"])
def receive_client_update():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"status": "error", "message": "Invalid payload"}), 400

    try:
        features = np.array(data["features"])
        client_updates.append(features)
        print(f"[INFO] Received update #{len(client_updates)}")

        # Once we collect enough updates, perform aggregation
        if len(client_updates) >= 3:  # threshold for federated round
            global_features = simple_aggregate(client_updates)
            np.save(AGGREGATED_DATA_PATH, global_features)
            with open(GLOBAL_MODEL_PATH, "wb") as f:
                pickle.dump(global_features, f)
            print("[INFO] Aggregated model updated.")
            client_updates.clear()

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/get_model", methods=["GET"])
def send_model():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        return jsonify({"status": "error", "message": "Model not available yet."}), 404

    with open(GLOBAL_MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
        return jsonify({"status": "success", "model": model_data.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
