# train_model.py
import os, json, joblib, numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from data_prep import prepare_and_split, ARTIFACT_DIR
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_model.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")

def train(data_path="data/gmsc_credit_data.csv"):
    X_train, X_test, y_train, y_test, _, _ = prepare_and_split(data_path)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    acc = float(accuracy_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {"accuracy": acc, "auc": auc, "classification_report": report, "confusion_matrix": cm}
    with open(METRICS_PATH, "w") as f: json.dump(metrics, f, indent=2)
    plt.figure(figsize=(5,4))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"))
    return model, metrics

if __name__ == "__main__":
    train()
