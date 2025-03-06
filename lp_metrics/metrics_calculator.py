import pandas as pd
from typing import Dict, Tuple

def calculate_metrics(ufm_dict, sorted_results_dict, conf_threshold):
    """Calculate performance metrics at specified confidence threshold"""
    tp = fp = fn = tn = 0
    manual = auto = 0
    missing = present = 0

    for ufm_id, vals in ufm_dict.items():
        if ufm_id not in sorted_results_dict:
            missing += 1
            continue
        else:
            present += 1
        lp_label = vals["PLATE_READ"]
        lp_pred = sorted_results_dict[ufm_id][1]
        lp_conf = sorted_results_dict[ufm_id][2]
        if lp_conf < conf_threshold:
            manual += 1
            if lp_label == lp_pred:
                fn += 1
            else:
                tn += 1
        elif lp_label == lp_pred:  # confidence >= 900
            auto += 1
            tp += 1
        else:
            auto += 1
            fp += 1
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return {
        'accuracy': (tp + tn) / (tp + tn + fp + fn + 1e-6),
        'precision': precision,
        'recall': recall,
        'f1': (2*precision*recall)/(precision+recall+1e-6),
        'automation_rate': auto / (auto + manual),
        'missing_ratio': missing / (missing + present)
    }