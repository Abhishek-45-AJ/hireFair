import numpy as np
from sklearn.metrics import accuracy_score

def compute_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    Calculates key algorithmic fairness metrics:
    - Selection Rate (SR) for Male and Female
    - Demographic Parity Difference |SR_Male - SR_Female|
    - True Positive Rate (TPR) for Male and Female
    - Equal Opportunity Difference |TPR_Male - TPR_Female|
    - Task Accuracy
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    sens = np.array(sensitive_attr).flatten()
    
    mask_m = (sens == 0)
    mask_f = (sens == 1)
    
    # Selection Rates with zero-denominator protection
    sr_m = float(np.mean(y_pred[mask_m])) if np.sum(mask_m) > 0 else 0.0
    sr_f = float(np.mean(y_pred[mask_f])) if np.sum(mask_f) > 0 else 0.0
    dp_diff = abs(sr_m - sr_f)
    
    # True Positive Rates (Equal Opportunity)
    pos_m = (y_true[mask_m] == 1)
    tp_m = np.sum((y_pred[mask_m] == 1) & pos_m)
    actual_pos_m = np.sum(pos_m)
    tpr_m = float(tp_m / actual_pos_m) if actual_pos_m > 0 else 0.0
    
    pos_f = (y_true[mask_f] == 1)
    tp_f = np.sum((y_pred[mask_f] == 1) & pos_f)
    actual_pos_f = np.sum(pos_f)
    tpr_f = float(tp_f / actual_pos_f) if actual_pos_f > 0 else 0.0
    
    eopp_diff = abs(tpr_m - tpr_f)
    acc = float(accuracy_score(y_true, y_pred))
    
    return {
        'selection_rate_male': round(sr_m, 4),
        'selection_rate_female': round(sr_f, 4),
        'demographic_parity_difference': round(dp_diff, 4),
        'tpr_male': round(tpr_m, 4),
        'tpr_female': round(tpr_f, 4),
        'equal_opportunity_difference': round(eopp_diff, 4),
        'accuracy': round(acc, 4)
    }