"""Automated model quality gate for deployment eligibility."""

from typing import Any

def passes_quality_gate(
    metrics: dict[str, Any],
    split_name: str = "validation",
    max_wape: float = 0.20,
) -> tuple[bool, str]:
    """
    Check if the candidate model meets production criteria:
    1. Candidate MAE < Previous Day MAE
    2. Candidate MAE < Previous Weekday MAE
    3. Candidate WAPE <= max_wape
    """
    if split_name not in metrics:
        return False, f"Missing {split_name} metrics."
        
    split_metrics = metrics[split_name]
    
    if "candidate" not in split_metrics:
        return False, "Missing candidate metrics."
        
    candidate_mae = split_metrics["candidate"]["mae"]
    candidate_wape = split_metrics["candidate"]["wape"]
    
    # Baseline checks
    for baseline in ["previous_day", "previous_weekday"]:
        if baseline in split_metrics:
            baseline_mae = split_metrics[baseline]["mae"]
            if candidate_mae >= baseline_mae:
                return False, f"Candidate MAE ({candidate_mae:.2f}) does not beat {baseline} ({baseline_mae:.2f})."
                
    # Absolute quality check
    if candidate_wape > max_wape:
        return False, f"Candidate WAPE ({candidate_wape:.2%}) exceeds threshold ({max_wape:.2%})."
        
    return True, "Passed all quality checks."
