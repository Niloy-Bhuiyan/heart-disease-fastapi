import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "heart_model.joblib")

def load_model():
    model = joblib.load(MODEL_PATH)
    return model
