# src/predictors/tsunami.py
from typing import Dict

def predict_tsunami(magnitude: float, depth: float, distance_to_coast_km: float) -> Dict:
    mag = float(magnitude)
    depth = float(depth)
    dist = float(distance_to_coast_km)
    risk_score = 0.0
    if mag >= 6.5:
        risk_score += 0.6
    if depth <= 70:
        risk_score += 0.25
    if dist <= 200:
        risk_score += 0.2
    prob = min(1.0, round(risk_score, 3))
    if prob >= 0.7:
        risk = "high"
    elif prob >= 0.35:
        risk = "medium"
    else:
        risk = "low"
    return {
        "tsunami_probability": prob,
        "risk": risk,
        "magnitude": mag,
        "depth": depth,
        "distance_to_coast_km": dist
    }
