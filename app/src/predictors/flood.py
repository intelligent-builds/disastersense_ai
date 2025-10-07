# src/predictors/flood.py
from typing import Dict

def predict_flood(rainfall_mm: float, dam_capacity_percent: float = 50.0) -> Dict:
    r = float(rainfall_mm)
    cap = float(dam_capacity_percent)
    score = min(1.0, (r / 200.0) * 0.7 + (cap / 100.0) * 0.5)
    prob = round(score, 3)
    reason_parts = []
    if r > 100:
        reason_parts.append("heavy rainfall")
    if cap > 80:
        reason_parts.append("dam near capacity")
    if not reason_parts:
        reason_parts.append("moderate conditions")
    return {
        "flood_probability": prob,
        "reason": ", ".join(reason_parts),
        "rainfall_mm": r,
        "dam_capacity_percent": cap
    }
