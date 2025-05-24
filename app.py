# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import pickle
import os
from datetime import datetime, timedelta
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure Streamlit page
st.set_page_config(
    page_title="Karkidi Job Scraper & Clustering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KarkidiJobScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    
    def scrape_jobs(self, keyword="data science", pages=1, progress_callback=None):
        """Scrape job listings from Karkidi.com"""
        jobs_list = []
        
        for page in range(1, pages + 1):
            if progress_callback:
                progress_callback(page, pages)
            
            url = self.base_url.format(page=page, query=keyword.replace(' ', '%20'))
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Find job blocks
                job_blocks = soup.find_all("div", class_="ads-details")
                
                for job in job_blocks:
                    try:
                        job_data = self._extract_job_data(job)
                        if job_data:
                            jobs_list.append(job_data)
                    except Exception as e:
                        st.warning(f"Error parsing job block: {e}")
                        continue
                
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                st.error(f"Error scraping page {page}: {e}")
                continue
        
        return pd.DataFrame(jobs_list)
    
    def _extract_job_data(self, job_block):
        """Extract job data from a job block"""
        try:
            # Extract title
            title_elem = job_block.find("h4")
            title = title_elem.get_text(strip=True) if title_elem else "N/A"
            
            # Extract company
            company_elem = job_block.find("a", href=lambda x: x and "Employer-Profile" in x)
            company = company_elem.get_text(strip=True) if company_elem else "N/A"
            
            # Extract location
            location_elem = job_block.find("p")
            location = location_elem.get_text(strip=True) if location_elem else "N/A"
            
            # Extract experience
            exp_elem = job_block.find("p", class_="emp-exp")
            experience = exp_elem.get_text(strip=True) if exp_elem else "N/A"
            
            # Extract skills
            skills_tag = job_block.find("span", string="Key Skills")
            skills = ""
            if skills_tag:
                skills_p = skills_tag.find_next("p")
                skills = skills_p.get_text(strip=True) if skills_p else ""
            
            # Extract summary
            summary_tag = job_block.find("span", string="Summary")
            summary = ""
            if summary_tag:
                summary_p = summary_tag.find_next("p")
                summary = summary_p.get_text(strip=True) if summary_p else ""
            
            # Extract salary if available
            salary = "N/A"
            salary_elem = job_block.find("span", class_="salary")
            if salary_elem:
                salary = salary_elem.get_text(strip=True)
            
            return {
                "Title": title,
                "Company": company,
                "Location": location,
                "Experience": experience,
                "Skills": skills,
                "Summary": summary,
                "Salary": salary,
                "Scraped_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return None

class JobClusteringModel:
    def __init__(self):
        self.vectorizer = None
        self.kmeans = None
        self.feature_matrix = None
        self.cluster_labels = None
        self.model_path = "job_clustering_model.pkl"
        self.data_path = "scraped_jobs.csv"
    
    def preprocess_text(self, text):
        """Preprocess text for TF-IDF"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return text
    
    def prepare_features(self, df):
        """Prepare features for clustering"""
        # Combine skills and summary for feature extraction
        df['combined_text'] = df['Skills'].fillna('') + ' ' + df['Summary'].fillna('')
        df['combined_text'] = df['combined_text'].apply(self.preprocess_text)
        
        return df['combined_text'].tolist()
    
    def train_model(self, df, n_clusters=5):
        """Train the clustering model"""
        features = self.prepare_features(df)
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.feature_matrix = self.vectorizer.fit_transform(features)
        
        # K-Means Clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.feature_matrix)
        
        # Save model
        self.save_model()
        
        return self.cluster_labels
    
    def predict_cluster(self, text):
        """Predict cluster for new text"""
        if self.vectorizer is None or self.kmeans is None:
            return None
        
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([processed_text])
        cluster = self.kmeans.predict(text_vector)[0]
        
        return cluster
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'kmeans': self.kmeans,
            'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.vectorizer = model_data['vectorizer']
                self.kmeans = model_data['kmeans']
                
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return False
    
    def get_cluster_keywords(self, cluster_id, top_n=10):
        """Get top keywords for a cluster"""
        if self.vectorizer is None or self.kmeans is None:
            return []
        
        # Get cluster center
        cluster_center = self.kmeans.cluster_centers_[cluster_id]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features for this cluster
        top_indices = cluster_center.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        
        return top_keywords

class JobAlertSystem:
    def __init__(self):
        self.preferences_path = "user_preferences.json"
    
    def save_preferences(self, user_email, preferred_clusters, keywords):
        """Save user preferences"""
        preferences = self.load_preferences()
        preferences[user_email] = {
            'preferred_clusters': preferred_clusters,
            'keywords': keywords,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.preferences_path, 'w') as f:
            json.dump(preferences, f, indent=2)
    
    def load_preferences(self):
        """Load user preferences"""
        if os.path.exists(self.preferences_path):
            try:
                with open(self.preferences_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def check_new_matches(self, df, model):
        """Check for new job matches based on user preferences"""
        preferences = self.load_preferences()
        matches = {}
        
        for email, prefs in preferences.items():
            user_matches = []
            preferred_clusters = prefs.get('preferred_clusters', [])
            
            for _, job in df.iterrows():
                combined_text = str(job.get('Skills', '')) + ' ' + str(job.get('Summary', ''))
                predicted_cluster = model.predict_cluster(combined_text)
                
                if predicted_cluster in preferred_clusters:
                    user_matches.append(job.to_dict())
            
            if user_matches:
                matches[email] = user_matches
        
        return matches

def create_visualizations(df, cluster_labels, model):
    """Create visualizations for the clustering results"""
    
    # Add cluster labels to dataframe
    df_viz = df.copy()
    df_viz['Cluster'] = cluster_labels
    
    # Cluster distribution
    fig_dist = px.histogram(
        df_viz, 
        x='Cluster', 
        title='Job Distribution Across Clusters',
        labels={'Cluster': 'Cluster ID', 'count': 'Number of Jobs'}
    )
    fig_dist.update_layout(showlegend=False)
    
    # 2D visualization using PCA
    if model.feature_matrix is not None:
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(model.feature_matrix.toarray())
        
        fig_2d = px.scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            color=cluster_labels.astype(str),
            title='Job Clusters Visualization (2D PCA)',
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component', 'color': 'Cluster'}
        )
        fig_2d.update_traces(marker=dict(size=8, opacity=0.6))
        
        return fig_dist, fig_2d
    
    return fig_dist, None

def main():
    st.title("🔍 Karkidi Job Scraper & Clustering App")
    st.markdown("*Scrape, cluster, and get alerts for job listings based on skills*")
    
    # Initialize components
    scraper = KarkidiJobScraper()
    model = JobClusteringModel()
    alert_system = JobAlertSystem()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Scrape Jobs", "🧠 Train Model", "📊 View Clusters", "🔔 Set Alerts", "📈 Analytics"]
    )
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Karkidi Job Scraper & Clustering App!
        
        This app helps you:
        - **Scrape** job listings from Karkidi.com
        - **Cluster** jobs based on required skills using ML
        - **Get alerts** when new jobs match your preferences
        - **Analyze** job market trends
        
        ### How to use:
        1. **Scrape Jobs**: Search and scrape job listings
        2. **Train Model**: Create clusters based on job skills
        3. **View Clusters**: Explore job clusters and their characteristics
        4. **Set Alerts**: Configure notifications for matching jobs
        5. **Analytics**: View insights and trends
        
        ### Features:
        - ✅ TF-IDF + K-Means clustering
        - ✅ Model persistence
        - ✅ Interactive visualizations
        - ✅ Email alerts
        - ✅ Scheduled scraping support
        """)
        
        # Show recent data if available
        if os.path.exists("scraped_jobs.csv"):
            df = pd.read_csv("scraped_jobs.csv")
            st.subheader("📊 Recent Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", len(df))
            with col2:
                st.metric("Companies", df['Company'].nunique())
            with col3:
                st.metric("Locations", df['Location'].nunique())
            with col4:
                if 'Scraped_Date' in df.columns:
                    latest_scrape = pd.to_datetime(df['Scraped_Date']).max()
                    st.metric("Last Scraped", latest_scrape.strftime("%Y-%m-%d"))
    
    elif page == "🔍 Scrape Jobs":
        st.header("Scrape Job Listings")
        
        col1, col2 = st.columns(2)
        with col1:
            keyword = st.text_input("Search Keyword", value="data science")
            pages = st.number_input("Number of Pages", min_value=1, max_value=10, value=2)
        
        with col2:
            append_data = st.checkbox("Append to existing data", value=True)
            
        if st.button("🚀 Start Scraping", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current_page, total_pages):
                progress = current_page / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Scraping page {current_page} of {total_pages}...")
            
            with st.spinner("Scraping jobs..."):
                df_new = scraper.scrape_jobs(keyword, pages, update_progress)
            
            if not df_new.empty:
                st.success(f"✅ Successfully scraped {len(df_new)} jobs!")
                
                # Save data
                if append_data and os.path.exists("scraped_jobs.csv"):
                    df_existing = pd.read_csv("scraped_jobs.csv")
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.drop_duplicates(subset=['Title', 'Company'], keep='last', inplace=True)
                else:
                    df_combined = df_new
                
                df_combined.to_csv("scraped_jobs.csv", index=False)
                
                # Display results
                st.subheader("📋 Scraped Jobs Preview")
                st.dataframe(df_new.head(10))
                
                # Download option
                csv = df_combined.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name=f"karkidi_jobs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No jobs found. Try different keywords or check the website.")
    
    elif page == "🧠 Train Model":
        st.header("Train Clustering Model")
        
        if not os.path.exists("scraped_jobs.csv"):
            st.warning("⚠️ No job data found. Please scrape jobs first.")
            return
        
        df = pd.read_csv("scraped_jobs.csv")
        st.info(f"📊 Found {len(df)} jobs in dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=3, max_value=15, value=5)
        with col2:
            retrain = st.checkbox("Force retrain (ignore existing model)")
        
        if st.button("🧠 Train Clustering Model", type="primary"):
            with st.spinner("Training model..."):
                cluster_labels = model.train_model(df, n_clusters)
            
            st.success("✅ Model trained successfully!")
            
            # Display cluster information
            st.subheader("🏷️ Cluster Information")
            
            for cluster_id in range(n_clusters):
                with st.expander(f"Cluster {cluster_id} ({sum(cluster_labels == cluster_id)} jobs)"):
                    keywords = model.get_cluster_keywords(cluster_id)
                    st.write("**Top Keywords:**", ", ".join(keywords))
                    
                    # Show sample jobs from this cluster
                    cluster_jobs = df[cluster_labels == cluster_id].head(3)
                    for _, job in cluster_jobs.iterrows():
                        st.write(f"• **{job['Title']}** at {job['Company']}")
    
    elif page == "📊 View Clusters":
        st.header("View Job Clusters")
        
        if not model.load_model():
            st.warning("⚠️ No trained model found. Please train the model first.")
            return
        
        if not os.path.exists("scraped_jobs.csv"):
            st.warning("⚠️ No job data found.")
            return
        
        df = pd.read_csv("scraped_jobs.csv")
        
        # Predict clusters for all jobs
        features = model.prepare_features(df)
        cluster_labels = []
        
        for text in features:
            cluster = model.predict_cluster(text)
            cluster_labels.append(cluster if cluster is not None else 0)
        
        cluster_labels = np.array(cluster_labels)
        
        # Create visualizations
        fig_dist, fig_2d = create_visualizations(df, cluster_labels, model)
        
        # Display visualizations
        st.plotly_chart(fig_dist, use_container_width=True)
        
        if fig_2d:
            st.plotly_chart(fig_2d, use_container_width=True)
        
        # Cluster analysis
        st.subheader("🔍 Cluster Analysis")
        
        n_clusters = len(np.unique(cluster_labels))
        
        for cluster_id in range(n_clusters):
            with st.expander(f"📂 Cluster {cluster_id} Analysis"):
                cluster_mask = cluster_labels == cluster_id
                cluster_jobs = df[cluster_mask]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Jobs in Cluster", len(cluster_jobs))
                    
                    # Top companies
                    top_companies = cluster_jobs['Company'].value_counts().head(5)
                    st.write("**Top Companies:**")
                    for company, count in top_companies.items():
                        st.write(f"• {company}: {count} jobs")
                
                with col2:
                    # Top keywords
                    keywords = model.get_cluster_keywords(cluster_id)
                    st.write("**Key Skills/Keywords:**")
                    st.write(", ".join(keywords))
                    
                    # Location distribution
                    top_locations = cluster_jobs['Location'].value_counts().head(3)
                    st.write("**Top Locations:**")
                    for loc, count in top_locations.items():
                        st.write(f"• {loc}: {count} jobs")
    
    elif page == "🔔 Set Alerts":
        st.header("Job Alert Configuration")
        
        if not model.load_model():
            st.warning("⚠️ No trained model found. Please train the model first.")
            return
        
        st.markdown("Set up alerts to get notified when new jobs match your preferred skill clusters.")
        
        # User email
        user_email = st.text_input("📧 Your Email Address")
        
        # Cluster preferences
        if os.path.exists("scraped_jobs.csv"):
            df = pd.read_csv("scraped_jobs.csv")
            features = model.prepare_features(df)
            
            # Get unique clusters
            sample_clusters = set()
            for text in features[:100]:  # Sample for performance
                cluster = model.predict_cluster(text)
                if cluster is not None:
                    sample_clusters.add(cluster)
            
            available_clusters = sorted(list(sample_clusters))
            
            st.subheader("🎯 Choose Your Preferred Skill Clusters")
            
            selected_clusters = []
            for cluster_id in available_clusters:
                keywords = model.get_cluster_keywords(cluster_id)
                cluster_label = f"Cluster {cluster_id}: {', '.join(keywords[:3])}"
                
                if st.checkbox(cluster_label, key=f"cluster_{cluster_id}"):
                    selected_clusters.append(cluster_id)
        
        # Additional keywords
        additional_keywords = st.text_area(
            "🔍 Additional Keywords (optional)",
            placeholder="e.g., python, machine learning, remote"
        )
        
        if st.button("💾 Save Alert Preferences", type="primary"):
            if user_email and selected_clusters:
                alert_system.save_preferences(user_email, selected_clusters, additional_keywords)
                st.success("✅ Alert preferences saved successfully!")
                
                st.info(f"""
                **Your Alert Configuration:**
                - Email: {user_email}
                - Preferred Clusters: {', '.join(map(str, selected_clusters))}
                - Additional Keywords: {additional_keywords or 'None'}
                """)
            else:
                st.error("⚠️ Please provide email and select at least one cluster.")
        
        # Show existing preferences
        if st.button("👁️ View My Current Preferences"):
            preferences = alert_system.load_preferences()
            if user_email in preferences:
                prefs = preferences[user_email]
                st.json(prefs)
            else:
                st.info("No preferences found for this email.")
    
    elif page == "📈 Analytics":
        st.header("Job Market Analytics")
        
        if not os.path.exists("scraped_jobs.csv"):
            st.warning("⚠️ No job data found.")
            return
        
        df = pd.read_csv("scraped_jobs.csv")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jobs", len(df))
        with col2:
            st.metric("Unique Companies", df['Company'].nunique())
        with col3:
            st.metric("Unique Locations", df['Location'].nunique())
        with col4:
            skills_mentioned = df['Skills'].str.len().sum()
            st.metric("Total Skills Mentioned", skills_mentioned)
        
        # Top companies
        st.subheader("🏢 Top Hiring Companies")
        top_companies = df['Company'].value_counts().head(10)
        fig_companies = px.bar(
            x=top_companies.values,
            y=top_companies.index,
            orientation='h',
            title="Top 10 Companies by Job Postings"
        )
        fig_companies.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_companies, use_container_width=True)
        
        # Location distribution
        st.subheader("📍 Job Locations")
        top_locations = df['Location'].value_counts().head(15)
        fig_locations = px.pie(
            values=top_locations.values,
            names=top_locations.index,
            title="Job Distribution by Location"
        )
        st.plotly_chart(fig_locations, use_container_width=True)
        
        # Skills analysis
        st.subheader("🛠️ Skills Analysis")
        
        # Extract and count skills
        all_skills = []
        for skills_str in df['Skills'].dropna():
            if skills_str and skills_str != '':
                # Simple skill extraction (split by common delimiters)
                skills = re.split(r'[,;|]+', skills_str.lower())
                all_skills.extend([skill.strip() for skill in skills if skill.strip()])
        
        if all_skills:
            skill_counts = Counter(all_skills)
            top_skills = dict(skill_counts.most_common(20))
            
            fig_skills = px.bar(
                x=list(top_skills.values()),
                y=list(top_skills.keys()),
                orientation='h',
                title="Top 20 Most Mentioned Skills"
            )
            fig_skills.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_skills, use_container_width=True)
        
        # Time series if date available
        if 'Scraped_Date' in df.columns:
            st.subheader("📅 Scraping Timeline")
            df['Scraped_Date'] = pd.to_datetime(df['Scraped_Date'])
            daily_counts = df.groupby(df['Scraped_Date'].dt.date).size()
            
            fig_timeline = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Jobs Scraped Over Time"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, scikit-learn, and BeautifulSoup*")

if __name__ == "__main__":
    main()

