import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

kmeans = joblib.load("job_cluster_model.pkl")
vectorizer = joblib.load("skill_vectorizer.pkl")

def preprocess_skills(skill_str):
    if pd.isna(skill_str): return ""
    return ','.join([s.strip().lower() for s in skill_str.split(',') if s.strip()])

def classify_job(skills_text):
    skills_text = preprocess_skills(skills_text)
    vectorized = vectorizer.transform([skills_text])
    return int(kmeans.predict(vectorized)[0])
