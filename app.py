# app.py

import streamlit as st
import pandas as pd
import requests
import os
import joblib
import time
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "kmeans_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "jobs_data.csv"

st.set_page_config(page_title="Karkidi Job Monitor", layout="wide")
st.title("🧠 Karkidi Job Monitor with Skill Clustering")

# 1. Scraper
def scrape_karkidi_jobs(keyword="data science", pages=2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            st.error(f"Error loading page {page}: {e}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills,
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            except:
                continue
        time.sleep(1)

    df = pd.DataFrame(jobs_list)
    if not df.empty:
        df.to_csv(DATA_PATH, index=False)
    return df

# 2. Training / Loading
def comma_tokenizer(text):
    return text.split(',')

def prepare_and_cluster(df, n_clusters=5):
    df['Cleaned_Skills'] = df['Skills'].fillna("").apply(lambda x: ','.join([s.strip().lower() for s in x.split(',')]))
    vectorizer = TfidfVectorizer(tokenizer=comma_tokenizer, lowercase=True, stop_words='english')
    X = vectorizer.fit_transform(df['Cleaned_Skills'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X)

    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return df


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        kmeans = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return kmeans, vectorizer
    else:
        return None, None

# 3. UI Inputs
with st.sidebar:
    st.header("🔧 Settings")
    keyword = st.text_input("Job Keyword", "data science")
    pages = st.slider("Pages to Scrape", 1, 5, 2)
    user_skills = st.text_input("Your Skills (comma-separated)", "python, machine learning, SQL")
    cluster_count = st.slider("Number of Clusters", 2, 10, 5)

# 4. Main Logic
if st.button("🚀 Run Scraper and Cluster"):
    with st.spinner("Scraping job postings..."):
        df_jobs = scrape_karkidi_jobs(keyword, pages)

    if df_jobs.empty:
        st.warning("No jobs found.")
    else:
        with st.spinner("Clustering jobs..."):
            clustered_df = prepare_and_cluster(df_jobs, cluster_count)
            st.success("Model trained and saved!")

        kmeans, vectorizer = load_model()
        if kmeans and vectorizer:
            user_vec = vectorizer.transform([user_skills.lower()])
            user_cluster = kmeans.predict(user_vec)[0]

            st.subheader(f"🎯 Jobs matching your skills (Cluster {user_cluster})")
            matches = clustered_df[clustered_df["Cluster"] == user_cluster]
            st.dataframe(matches[["Title", "Company", "Location", "Skills"]], use_container_width=True)

            with st.expander("📊 View All Jobs with Clusters"):
                st.dataframe(clustered_df[["Title", "Company", "Location", "Skills", "Cluster"]], use_container_width=True)

            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 5. Alert User if New Job Matches Cluster (Optional)
if st.button("🔔 Check for New Matching Jobs"):
    try:
        old_df = pd.read_csv(DATA_PATH)
        kmeans, vectorizer = load_model()
        new_df = scrape_karkidi_jobs(keyword, pages)

        if new_df.empty or kmeans is None or vectorizer is None:
            st.warning("Insufficient data to compare.")
        else:
            new_df['Cleaned_Skills'] = new_df['Skills'].fillna("").apply(lambda x: ','.join([s.strip().lower() for s in x.split(',')]))
            X_new = vectorizer.transform(new_df['Cleaned_Skills'])
            new_df['Cluster'] = kmeans.predict(X_new)

            user_vec = vectorizer.transform([user_skills.lower()])
            user_cluster = kmeans.predict(user_vec)[0]

            new_cluster_matches = new_df[new_df["Cluster"] == user_cluster]

            # Check if any jobs in new matches are not in old
            merged = new_cluster_matches.merge(old_df, on=["Title", "Company"], how="left", indicator=True)
            unseen_jobs = merged[merged["_merge"] == "left_only"]

            if unseen_jobs.empty:
                st.info("No new matching jobs.")
            else:
                st.success(f"🔔 {len(unseen_jobs)} new jobs match your skills!")
                st.dataframe(unseen_jobs[["Title", "Company", "Location", "Skills"]], use_container_width=True)
    except Exception as e:
        st.error(f"Error during alert check: {e}")


