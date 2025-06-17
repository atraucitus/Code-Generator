import json
import os

def load_conala(path="conala-train.json", max_samples=5000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Extract NL-code pairs
    data = [(item["intent"], item["snippet"]) for item in raw if "intent" in item and "snippet" in item]
    
    return data[:max_samples]
