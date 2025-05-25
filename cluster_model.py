# # cluster_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
from scraper import scrape_karkidi_jobs  # Make sure this import is valid

def preprocess_skills(skills_series):
    return skills_series.fillna("").str.lower().str.replace(",", " ").str.replace("  ", " ", regex=False)
    

def find_optimal_k(skill_vectors, max_k=10):
    sse = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(skill_vectors)
        sse.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure()
    plt.plot(range(2, max_k + 1), sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig("elbow_plot.png")  # Save the plot as an image file

    # Ask user to input optimal k (optional) or choose automatically
    optimal_k = int(input("Enter the optimal number of clusters based on the Elbow Plot (e.g., 5): "))
    return optimal_k

def cluster_jobs(df):
    df = df.copy()
    skills_processed = preprocess_skills(df["Skills"])

    vectorizer = TfidfVectorizer()
    skill_vectors = vectorizer.fit_transform(skills_processed)

    # Determine optimal number of clusters
    k = find_optimal_k(skill_vectors, max_k=10)

    # Fit KMeans with selected k
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(skill_vectors)

    df["Cluster"] = clusters

    # Evaluate clusters using silhouette score
    sil_score = silhouette_score(skill_vectors, clusters)
    print(f"Silhouette Score for k={k}: {sil_score:.4f}")

    # Save model components
    joblib.dump({
        "model": model,
        "vectorizer": vectorizer,
        "vectors": skill_vectors.toarray(),
        "labels": clusters
    }, "job_model.pkl")

    return df

if __name__ == "__main__":
    print("Scraping job data...")
    df = scrape_karkidi_jobs(keyword="data science", pages=6)

    print("Clustering jobs...")
    clustered_df = cluster_jobs(df)

    print("Saving clustered job data...")
    clustered_df.to_csv("clustered_jobs.csv", index=False)
    print("Done! Saved model to 'job_model.pkl' and job data to 'clustered_jobs.csv'")
