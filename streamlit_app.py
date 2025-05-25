
import streamlit as st
import pandas as pd
import json
import joblib
from cluster_model import preprocess_skills
from scraper import scrape_karkidi_jobs

# Load model
model_data = joblib.load("job_model.pkl")
model = model_data["model"]
vectorizer = model_data["vectorizer"]

# Cluster label descriptions
cluster_labels = { 
    0: "Data Analysis, Python, SQL",
    1: "Machine Learning, Scikit-learn, TensorFlow",
    2: "Data Engineering, ETL, Spark",
    3: "Web Development, JavaScript, React",
    4: "DevOps, AWS, Docker",
    5: "Cybersecurity, Network Security",
    6: "Business Intelligence, Power BI, Tableau",
    7: "Natural Language Processing, Transformers"
}
label_to_cluster = {v: k for k, v in cluster_labels.items()}

# Title
st.title("🤖🧠 Skill-Based Job Recommender System")

# User input
email = st.text_input("Enter your email")

selected_labels = st.multiselect(
    "Select your interested clusters",
    options=list(cluster_labels.values())
)

if st.button("Fetch Jobs"):
    if email and selected_labels:
        # Scrape and process jobs
        df = scrape_karkidi_jobs("data science", pages=3)
        df["Skills"] = preprocess_skills(df["Skills"])
        df["Cluster"] = model.predict(vectorizer.transform(df["Skills"]))

        # Filter by selected cluster labels
        selected_clusters = [label_to_cluster[label] for label in selected_labels]
        filtered_jobs = df[df["Cluster"].isin(selected_clusters)]

        # Display results
        if not filtered_jobs.empty:
            st.write(f"### Recommended Jobs for {email}")
            for idx, row in filtered_jobs.iterrows():
                skills_list = row["Skills"].split()
                formatted_skills = "\n".join([f"- {skill}" for skill in skills_list])
                st.markdown(f"""
**{row['Title']}** at *{row['Company']}*  
📍 **Location**: {row.get('Location', 'N/A')}  
🧠 **Skills**:  
{formatted_skills}  
📋 **Summary**: {row.get('Summary', 'No summary provided.')}  
---
""")
        else:
            st.warning("No matching jobs found.")
    else:
        st.warning("Please enter your email and select at least one cluster.")
