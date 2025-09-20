# explain_shap.py
import shap, joblib, os, json
import pandas as pd
from data_prep import ARTIFACT_DIR, PREPROCESSOR_PATH, FEATURES_PATH, BACKGROUND_PATH

MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_model.joblib")

def load_artifacts():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(PREPROCESSOR_PATH),
        joblib.load(FEATURES_PATH)
    )

def summary_shap():
    model, _, feature_names = load_artifacts()
    background = joblib.load(BACKGROUND_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(background)
    shap.summary_plot(shap_values, background, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(ARTIFACT_DIR, "shap_summary.png"))

def explain_instance(instance_raw: dict):
    model, preproc, feature_names = load_artifacts()
    df = pd.DataFrame([instance_raw])
    X_t = preproc.transform(df)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_t)
    vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
    feat_imp = sorted(zip(feature_names, vals.tolist()), key=lambda x: -abs(x[1]))[:10]
    return [{"feature": f, "shap_value": float(v)} for f, v in feat_imp]
