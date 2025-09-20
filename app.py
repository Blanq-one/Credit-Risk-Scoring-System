# api/app.py
from flask import Flask, request, jsonify
import joblib, os, pandas as pd
from explain_shap import explain_instance
from data_prep import ARTIFACT_DIR, PREPROCESSOR_PATH, FEATURES_PATH

MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_model.joblib")
app = Flask(__name__)

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = joblib.load(FEATURES_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if not payload: return jsonify({"error": "No JSON"}), 400
    try:
        df = pd.DataFrame([payload])
        X_t = preprocessor.transform(df)
        prob = float(model.predict_proba(X_t)[0,1])
        pred = int(model.predict(X_t)[0])
        explanation = explain_instance(payload)
        return jsonify({"predicted_label": pred, "default_probability": prob, "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
