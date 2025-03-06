import pandas as pd
from typing import Dict, Tuple

def calculate_metrics(
        merged_df: pd.DataFrame,
        confidence_threshold: int = 910,
) -> Dict[str, float]:
    
    """Calculate performance metrics at specified confidence threshold"""

    processed = preprocess_samples(merged_df)
    filtered = filter_preds(processed,confidence_threshold)
    return compute_metrics(filtered)

def preprocess_samples(merged_df: pd.DataFrame) -> pd.DataFrame:
    """"""
    return 0

def filter_preds():
    return 0

def compute_metrics():
    return 0