# fairness_checks.py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def disparate_impact(df, y_true_col, y_pred_col, sensitive_col, privileged_value):
    grp_priv = df[df[sensitive_col] == privileged_value]
    grp_unpriv = df[df[sensitive_col] != privileged_value]
    def positive_rate(g): return (g[y_pred_col] == 1).mean() if len(g)>0 else 0
    p_priv, p_unpriv = positive_rate(grp_priv), positive_rate(grp_unpriv)
    return float(p_unpriv / p_priv) if p_priv else np.nan

def tpr_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp+fn) if tp+fn else 0, fp/(fp+tn) if fp+tn else 0

def equalized_odds_summary(df, y_true_col, y_pred_col, sensitive_col, privileged_value):
    grp_priv = df[df[sensitive_col] == privileged_value]
    grp_unpriv = df[df[sensitive_col] != privileged_value]
    tpr_priv, fpr_priv = tpr_fpr(grp_priv[y_true_col], grp_priv[y_pred_col]) if len(grp_priv)>0 else (None,None)
    tpr_unpriv, fpr_unpriv = tpr_fpr(grp_unpriv[y_true_col], grp_unpriv[y_pred_col]) if len(grp_unpriv)>0 else (None,None)
    return {"privileged": {"TPR": tpr_priv, "FPR": fpr_priv}, "unprivileged": {"TPR": tpr_unpriv, "FPR": fpr_unpriv}}
