# src/preprocess.py
import re
import pandas as pd

def clean_text(text: str) -> str:
    """Minimal cleaning for incoming messages (no dataset-wide changes)."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+(\.\d+)?', ' number ', text)
    text = text.encode('ascii', errors='ignore').decode()
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text
