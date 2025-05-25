from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

def preprocess_skills(df):
    df['skills_clean'] = df['skills'].fillna('').str.lower().str.replace(',', ' ')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['skills_clean'])
    return X, vectorizer

def cluster_skills(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return model, labels

def save_model(model, vectorizer):
    joblib.dump(model, "model/job_cluster_model.pkl")
    joblib.dump(vectorizer, "model/skills_vectorizer.pkl")
