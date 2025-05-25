import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

def train_cluster_model(csv_path="jobs_latest.csv", num_clusters=5):
    df = pd.read_csv(csv_path)
    skills = df['Skills'].fillna("").values

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(skills)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    df['Cluster'] = kmeans.labels_

    # Save model, vectorizer, and updated data
    joblib.dump(kmeans, "kmeans_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    df.to_csv("jobs_latest.csv", index=False)
    print("Training done. Models and updated data saved.")

if __name__ == "__main__":
    train_cluster_model()
