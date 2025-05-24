import streamlit as st
import pandas as pd
from scraper import scrape_karkidi_jobs
from model_utils import classify_job

st.title("AI-Powered Job Monitor from Karkidi")

user_skills = st.text_input("Enter your skills (comma-separated):", "python, machine learning, SQL")

if st.button("Scrape and Classify Jobs"):
    with st.spinner("Scraping job listings..."):
        df = scrape_karkidi_jobs(keyword="data science", pages=2)
        df["Cleaned_Skills"] = df["Skills"].fillna("").apply(lambda x: ','.join([s.strip().lower() for s in x.split(',')]))
        df["Cluster"] = df["Cleaned_Skills"].apply(classify_job)

        st.success("Scraping and classification complete!")
        st.write(df[["Title", "Company", "Location", "Skills", "Cluster"]].head(10))

        user_cluster = classify_job(user_skills)
        matched_jobs = df[df["Cluster"] == user_cluster]

        st.subheader("🎯 Jobs Matching Your Skills")
        st.write(matched_jobs[["Title", "Company", "Location", "Skills"]].reset_index(drop=True))
