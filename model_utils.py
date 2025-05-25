# model_utils.py

import joblib

def save_model(vectorizer, model, path="job_model.pkl"):
    joblib.dump({"vectorizer": vectorizer, "model": model}, path)

def load_model(path="job_model.pkl"):
    return joblib.load(path)
