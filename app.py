import streamlit as st
import pandas as pd
from scraper import scrape_karkidi_jobs
from model import preprocess_skills, cluster_skills, save_model
import joblib
import os

st.set_page_config(page_title="Job Classifier", layout="wide")

st.title("💼 Job Posting Classifier")

menu = st.sidebar.selectbox("Choose Option", ["Train Model", "Classify New Jobs", "Job Alert"])

# --------------------- Train Model ---------------------
if menu == "Train Model":
    st.subheader("🧠 Train Skill-Based Clustering Model")
    keyword = st.text_input("Enter job keyword (e.g., data science):", "data science")
    pages = st.slider("Number of pages to scrape", 1, 5, 2)
    n_clusters = st.slider("Number of skill clusters", 2, 10, 5)

    if st.button("Scrape & Train"):
        df = scrape_karkidi_jobs(keyword=keyword, pages=pages)
        X, vectorizer = preprocess_skills(df)
        model, labels = cluster_skills(X, n_clusters)
        df['cluster'] = labels
        save_model(model, vectorizer)

        os.makedirs("output", exist_ok=True)
        df.to_csv("output/initial_clustered_jobs.csv", index=False)
        st.success(f"Model trained with {len(df)} jobs across {n_clusters} clusters.")
        st.dataframe(df.head())

# --------------------- Classify New Jobs ---------------------
elif menu == "Classify New Jobs":
    st.subheader("🆕 Classify Newly Scraped Jobs")

    if not os.path.exists("model/job_cluster_model.pkl"):
        st.warning("Train the model first in 'Train Model' tab.")
    else:
        if st.button("Scrape & Classify New Jobs"):
            df = scrape_karkidi_jobs(pages=1)
            model = joblib.load("model/job_cluster_model.pkl")
            vectorizer = joblib.load("model/skills_vectorizer.pkl")
            df['skills_clean'] = df['skills'].fillna('').str.lower().str.replace(',', ' ')
            X_new = vectorizer.transform(df['skills_clean'])
            df['cluster'] = model.predict(X_new)

            df.to_csv("output/daily_classified_jobs.csv", index=False)
            st.success("New jobs classified successfully.")
            st.dataframe(df.head())

# --------------------- Job Alert ---------------------
elif menu == "Job Alert":
    st.subheader("🔔 Job Alerts Based on Your Skills")
    user_keywords = st.text_input("Enter your preferred skill keywords (comma-separated)", "python, machine learning")

    if os.path.exists("output/daily_classified_jobs.csv"):
        df = pd.read_csv("output/daily_classified_jobs.csv")
        keywords = [k.strip().lower() for k in user_keywords.split(",")]

        matched_df = df[df['skills'].str.lower().apply(lambda skills: any(k in skills for k in keywords))]
        st.success(f"Found {len(matched_df)} matching jobs.")
        st.dataframe(matched_df[['title', 'company', 'skills', 'link']])
    else:
        st.warning("No daily classified jobs found. Run 'Classify New Jobs' first.")
