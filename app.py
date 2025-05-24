# app.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time

st.set_page_config(page_title="Karkidi Job Monitor", layout="wide")
st.title("🧠 Karkidi Job Monitor with Skill Clustering")

# 1. Web Scraper Function
@st.cache_data
def scrape_karkidi_jobs(keyword="data science", pages=2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
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
                    "Skills": skills
                })
            except:
                continue
        time.sleep(1)

    return pd.DataFrame(jobs_list)

# 2. Clustering Function
@st.cache_resource
def train_model(df, n_clusters=5):
    df['Cleaned_Skills'] = df['Skills'].fillna("").apply(lambda x: ','.join([s.strip().lower() for s in x.split(',')]))
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','), lowercase=True)
    X = vectorizer.fit_transform(df['Cleaned_Skills'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    df['Cluster'] = kmeans.predict(X)
    return df, kmeans, vectorizer

# 3. App Input and Processing
with st.sidebar:
    st.header("🔧 Settings")
    keyword = st.text_input("Job Keyword", "data science")
    pages = st.slider("Pages to Scrape", 1, 5, 2)
    user_skills = st.text_input("Your Skills (comma-separated)", "python, machine learning, SQL")

if st.button("🚀 Run Scraper and Cluster"):
    with st.spinner("Scraping job postings..."):
        df_jobs = scrape_karkidi_jobs(keyword, pages)
    
    if df_jobs.empty:
        st.warning("No jobs found.")
    else:
        with st.spinner("Clustering based on skills..."):
            clustered_df, model, vectorizer = train_model(df_jobs)

            # Predict user cluster
            user_vector = vectorizer.transform([user_skills.lower()])
            user_cluster = model.predict(user_vector)[0]

            st.success("Done!")
            st.subheader(f"🎯 Jobs matching your skills (Cluster {user_cluster})")
            matches = clustered_df[clustered_df["Cluster"] == user_cluster]
            st.dataframe(matches[["Title", "Company", "Location", "Skills"]].reset_index(drop=True), use_container_width=True)

            with st.expander("📊 View All Jobs with Cluster Assignments"):
                st.dataframe(clustered_df[["Title", "Company", "Location", "Skills", "Cluster"]].reset_index(drop=True), use_container_width=True)

