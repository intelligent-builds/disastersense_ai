# src/predictors/earthquake.py
from typing import Dict, Optional

def predict_quake(lat: float, lon: float, date: Optional[str]=None, depth: float=10.0) -> Dict:
    base_mag = 2.5
    depth_factor = max(0.0, (150.0 - float(depth)) / 150.0)
    pred_mag = base_mag + depth_factor * 4.0
    pred_mag = round(pred_mag, 2)
    if pred_mag >= 6.0:
        risk = "high"
    elif pred_mag >= 4.0:
        risk = "medium"
    else:
        risk = "low"
    return {
        "predicted_magnitude": pred_mag,
        "risk": risk,
        "depth_used": float(depth),
        "latitude": float(lat),
        "longitude": float(lon)
    }
