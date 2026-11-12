from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "claim_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = {
    "payer": str,
    "provider": str,
    "cpt_code": str,
    "diagnosis_code": str,
    "billed_amount": (int, float),
    "patient_age": int,
}


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    for field, expected in REQUIRED_FIELDS.items():
        value = payload[field]
        if expected is str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Field '{field}' must be a non-empty string.")
        elif expected in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Field '{field}' must be numeric.")
            if field == "billed_amount" and value <= 0:
                raise ValueError("Field 'billed_amount' must be greater than zero.")
            if field == "patient_age" and not (0 <= value <= 120):
                raise ValueError("Field 'patient_age' must be between 0 and 120.")

    return payload


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=False, silent=True)
        validated = _validate_payload(payload)

        row = pd.DataFrame(
            [{
                "payer": validated["payer"],
                "provider": validated["provider"],
                "cpt_code": validated["cpt_code"],
                "diagnosis_code": validated["diagnosis_code"],
                "billed_amount": float(validated["billed_amount"]),
                "patient_age": int(validated["patient_age"]),
            }]
        )

        predicted_status = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        labels = model.classes_

        response = {
            "predicted_status": predicted_status,
            "probabilities": {label: float(probabilities[idx]) for idx, label in enumerate(labels)},
        }
        return jsonify(response), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
