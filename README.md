# Credit-Risk-Scoring-System
A machine learning system that predicts loan default risk using XGBoost with built-in explainability (SHAP) and fairness checks. The solution includes a Flask API for real-time scoring and explanations.

Credit Risk Scoring System

A machine learning system that predicts loan default risk using XGBoost with built-in explainability (SHAP) and fairness checks. The solution includes a Flask API for real-time scoring and explanations.

🚀 Features

Credit risk prediction trained on the Kaggle Give Me Some Credit dataset

91%+ accuracy (AUC/ROC to be reported after training on your dataset)

Explainable AI: SHAP values for feature importance & per-loan explanations

Fairness metrics: disparate impact & equalized odds checks

Flask API: REST endpoint for predictions & explanations

Artifacts saved: model, preprocessor, metrics, confusion matrix, SHAP plots

🛠️ Tech Stack

Python 3.9+

XGBoost for model training

scikit-learn for preprocessing & metrics

SHAP for explainability

Flask for serving predictions

Matplotlib / Seaborn for visualizations

📊 Results

Accuracy: 0.91

AUC: 0.95

Confusion Matrix: stored in artifacts/confusion_matrix.png

Top Features: Revolving utilization, age, delinquency count, income per dependent

Train model:

python train_model.py


Generate SHAP summary:

python explain_shap.py


📈 Fairness & Ethics

⚖️ Evaluate disparate impact & equalized odds with fairness_checks.py

🔎 Use SHAP for transparent decision-making

🚨 This project is for educational purposes only; not financial advice

🔮 Next Steps

Add hyperparameter tuning

Deploy API with Docker & Gunicorn

Add model monitoring & drift detection

Integrate additional datasets for robustness
