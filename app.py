import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_data():
    return pd.read_csv("jobs_latest.csv")

@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

def main():
    st.title("Job Alert by Skill Clusters")

    user_skills = st.text_input("Enter your skills (comma separated)", "python, machine learning, data analysis")

    model, vectorizer = load_model()
    df = load_data()

    if st.button("Find Jobs"):
        user_vec = vectorizer.transform([user_skills])
        cluster_label = model.predict(user_vec)[0]

        matched_jobs = df[df['Cluster'] == cluster_label]

        st.write(f"Jobs matching your skill cluster ({cluster_label}):")
        if matched_jobs.empty:
            st.write("No matching jobs found.")
        else:
            for idx, row in matched_jobs.iterrows():
                st.write(f"**{row['Title']}** at {row['Company']} ({row['Location']})")
                st.write(f"Skills: {row['Skills']}")
                st.write(f"Experience: {row['Experience']}")
                st.write(f"Summary: {row['Summary']}")
                st.write("---")

if __name__ == "__main__":
    main()
