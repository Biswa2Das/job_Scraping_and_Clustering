# job_scraper_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
import sqlite3
import pickle
import joblib
from datetime import datetime, timedelta
import schedule
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json

# Configuration
DATABASE_FILE = "job_database.db"
MODEL_FILE = "job_clustering_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
USER_PREFERENCES_FILE = "user_preferences.json"

class JobScraper:
    """Handles web scraping of job postings from karkidi.com"""
    
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        self.base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    
    def scrape_jobs(self, keyword="data science", pages=2):
        """Scrape job postings from karkidi.com"""
        jobs_list = []
        
        for page in range(1, pages + 1):
            url = self.base_url.format(page=page, query=keyword.replace(' ', '%20'))
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                
                job_blocks = soup.find_all("div", class_="ads-details")
                
                for job in job_blocks:
                    try:
                        title = job.find("h4").get_text(strip=True) if job.find("h4") else "N/A"
                        
                        company_tag = job.find("a", href=lambda x: x and "Employer-Profile" in x)
                        company = company_tag.get_text(strip=True) if company_tag else "N/A"
                        
                        location_tag = job.find("p")
                        location = location_tag.get_text(strip=True) if location_tag else "N/A"
                        
                        exp_tag = job.find("p", class_="emp-exp")
                        experience = exp_tag.get_text(strip=True) if exp_tag else "N/A"
                        
                        key_skills_tag = job.find("span", string="Key Skills")
                        skills = ""
                        if key_skills_tag:
                            skills_p = key_skills_tag.find_next("p")
                            skills = skills_p.get_text(strip=True) if skills_p else ""
                        
                        summary_tag = job.find("span", string="Summary")
                        summary = ""
                        if summary_tag:
                            summary_p = summary_tag.find_next("p")
                            summary = summary_p.get_text(strip=True) if summary_p else ""
                        
                        # Create combined text for clustering
                        combined_text = f"{title} {skills} {summary}".lower()
                        
                        jobs_list.append({
                            "Title": title,
                            "Company": company,
                            "Location": location,
                            "Experience": experience,
                            "Summary": summary,
                            "Skills": skills,
                            "Combined_Text": combined_text,
                            "Post_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Keyword": keyword
                        })
                        
                    except Exception as e:
                        st.warning(f"Error parsing job block: {e}")
                        continue
                
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                st.error(f"Error scraping page {page}: {e}")
                continue
        
        return pd.DataFrame(jobs_list)

class DatabaseManager:
    """Handles SQLite database operations"""
    
    def __init__(self, db_file=DATABASE_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Jobs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            company TEXT,
            location TEXT,
            experience TEXT,
            summary TEXT,
            skills TEXT,
            combined_text TEXT,
            post_date TEXT,
            keyword TEXT,
            cluster_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # User preferences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            skills TEXT,
            preferred_clusters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_jobs(self, jobs_df):
        """Save jobs dataframe to database"""
        conn = sqlite3.connect(self.db_file)
        jobs_df.to_sql('jobs', conn, if_exists='append', index=False)
        conn.close()
    
    def get_jobs(self, limit=None):
        """Retrieve jobs from database"""
        conn = sqlite3.connect(self.db_file)
        query = "SELECT * FROM jobs ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        jobs_df = pd.read_sql_query(query, conn)
        conn.close()
        return jobs_df
    
    def clear_old_jobs(self, days=30):
        """Remove jobs older than specified days"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        cursor.execute("DELETE FROM jobs WHERE post_date < ?", (cutoff_date,))
        conn.commit()
        conn.close()

class JobClusterer:
    """Handles machine learning clustering of job postings"""
    
    def __init__(self):
        self.vectorizer = None
        self.clusterer = None
        self.scaler = StandardScaler()
        self.n_clusters = 5
    
    def preprocess_text(self, text_series):
        """Clean and preprocess text data"""
        processed_texts = []
        
        for text in text_series:
            if pd.isna(text) or text == "":
                processed_texts.append("")
                continue
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep alphanumeric and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            processed_texts.append(text)
        
        return processed_texts
    
    def vectorize_text(self, texts, fit=True):
        """Convert text to TF-IDF vectors"""
        if fit or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        best_score = -1
        best_k = 2
        
        for k in range(2, min(max_clusters + 1, len(X))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                continue
        
        return best_k
    
    def cluster_jobs(self, jobs_df, method='kmeans'):
        """Cluster jobs based on their text content"""
        if jobs_df.empty:
            return jobs_df, None
        
        # Preprocess text
        processed_texts = self.preprocess_text(jobs_df['Combined_Text'])
        
        # Vectorize text
        tfidf_matrix = self.vectorize_text(processed_texts, fit=True)
        
        # Convert to dense array for clustering
        X = tfidf_matrix.toarray()
        
        # Find optimal number of clusters
        self.n_clusters = self.find_optimal_clusters(X)
        
        # Perform clustering
        if method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=2)
        
        cluster_labels = self.clusterer.fit_predict(X)
        
        # Add cluster labels to dataframe
        jobs_df = jobs_df.copy()
        jobs_df['cluster_id'] = cluster_labels
        
        return jobs_df, self.get_cluster_summary(jobs_df)
    
    def get_cluster_summary(self, jobs_df):
        """Generate summary of each cluster"""
        cluster_summary = {}
        
        for cluster_id in jobs_df['cluster_id'].unique():
            if cluster_id == -1:  # DBSCAN noise points
                continue
            
            cluster_jobs = jobs_df[jobs_df['cluster_id'] == cluster_id]
            
            # Most common skills in this cluster
            all_skills = ' '.join(cluster_jobs['Skills'].fillna('').astype(str))
            skill_words = re.findall(r'\b\w+\b', all_skills.lower())
            common_skills = [word for word, count in Counter(skill_words).most_common(10) 
                           if len(word) > 2]
            
            # Most common job titles
            common_titles = cluster_jobs['Title'].value_counts().head(5).index.tolist()
            
            cluster_summary[cluster_id] = {
                'size': len(cluster_jobs),
                'common_skills': common_skills,
                'common_titles': common_titles,
                'avg_jobs_per_company': cluster_jobs.groupby('Company').size().mean()
            }
        
        return cluster_summary
    
    def predict_cluster(self, user_skills):
        """Predict which cluster a user belongs to based on their skills"""
        if self.vectorizer is None or self.clusterer is None:
            return None
        
        # Preprocess user skills
        processed_skills = self.preprocess_text([user_skills])[0]
        
        # Vectorize
        user_vector = self.vectorizer.transform([processed_skills])
        
        # Predict cluster
        cluster = self.clusterer.predict(user_vector.toarray())[0]
        
        return cluster
    
    def save_model(self):
        """Save trained model and vectorizer"""
        if self.clusterer is not None:
            joblib.dump(self.clusterer, MODEL_FILE)
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, VECTORIZER_FILE)
    
    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            if os.path.exists(MODEL_FILE):
                self.clusterer = joblib.load(MODEL_FILE)
            if os.path.exists(VECTORIZER_FILE):
                self.vectorizer = joblib.load(VECTORIZER_FILE)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

class NotificationManager:
    """Handles user notifications"""
    
    def __init__(self):
        self.user_preferences = self.load_user_preferences()
    
    def load_user_preferences(self):
        """Load user preferences from file"""
        if os.path.exists(USER_PREFERENCES_FILE):
            try:
                with open(USER_PREFERENCES_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_user_preferences(self):
        """Save user preferences to file"""
        with open(USER_PREFERENCES_FILE, 'w') as f:
            json.dump(self.user_preferences, f)
    
    def add_user_preference(self, email, skills):
        """Add user preference"""
        self.user_preferences[email] = {
            'skills': skills,
            'created_at': datetime.now().isoformat()
        }
        self.save_user_preferences()
    
    def send_email_notification(self, to_email, jobs_df, smtp_server=None, smtp_port=587, 
                              email_user=None, email_pass=None):
        """Send email notification about relevant jobs"""
        if not all([smtp_server, email_user, email_pass]):
            st.warning("Email configuration not set up")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = to_email
            msg['Subject'] = f"New Job Matches Found - {len(jobs_df)} Jobs"
            
            # Create HTML email body
            html_body = "<h2>New Job Matches</h2>"
            for _, job in jobs_df.iterrows():
                html_body += f"""
                <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
                    <h3>{job['Title']}</h3>
                    <p><strong>Company:</strong> {job['Company']}</p>
                    <p><strong>Location:</strong> {job['Location']}</p>
                    <p><strong>Skills:</strong> {job['Skills']}</p>
                </div>
                """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            st.error(f"Error sending email: {e}")
            return False

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all application components"""
    scraper = JobScraper()
    db_manager = DatabaseManager()
    clusterer = JobClusterer()
    notification_manager = NotificationManager()
    
    # Try to load existing model
    clusterer.load_model()
    
    return scraper, db_manager, clusterer, notification_manager

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Job Hunter Pro",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    scraper, db_manager, clusterer, notification_manager = init_components()
    
    # Sidebar navigation
    st.sidebar.title("🔍 Job Hunter Pro")
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Home", "📊 Job Dashboard", "🎯 Skill Matching", "⚙️ Admin Panel", "📧 Notifications"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Job Dashboard":
        show_job_dashboard(db_manager, clusterer)
    elif page == "🎯 Skill Matching":
        show_skill_matching(db_manager, clusterer, notification_manager)
    elif page == "⚙️ Admin Panel":
        show_admin_panel(scraper, db_manager, clusterer)
    elif page == "📧 Notifications":
        show_notifications_page(notification_manager)

def show_home_page():
    """Display home page"""
    st.title("🔍 Job Hunter Pro")
    st.markdown("### Your AI-Powered Job Discovery Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🚀 Features
        - **Smart Job Scraping**: Automatically scrape jobs from Karkidi.com
        - **AI Clustering**: Group jobs by required skills using machine learning
        - **Skill Matching**: Find jobs that match your specific skills
        - **Real-time Notifications**: Get notified about relevant new postings
        - **Interactive Dashboard**: Visualize job market trends
        """)
    
    with col2:
        st.markdown("""
        #### 📈 How It Works
        1. **Scrape**: We collect the latest job postings
        2. **Analyze**: AI clusters jobs by required skills
        3. **Match**: Input your skills to find relevant opportunities
        4. **Notify**: Get alerted when new matching jobs are posted
        5. **Apply**: Connect directly with employers
        """)
    
    st.markdown("---")
    
    # Quick stats
    try:
        jobs_df = db_manager.get_jobs(limit=1000)
        if not jobs_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", len(jobs_df))
            with col2:
                st.metric("Companies", jobs_df['Company'].nunique())
            with col3:
                st.metric("Locations", jobs_df['Location'].nunique())
            with col4:
                if 'cluster_id' in jobs_df.columns:
                    st.metric("Job Categories", jobs_df['cluster_id'].nunique())
                else:
                    st.metric("Job Categories", "Not clustered")
        else:
            st.info("No jobs in database yet. Go to Admin Panel to scrape some jobs!")
    except Exception as e:
        st.warning("Database not initialized yet. Please visit the Admin Panel first.")

def show_job_dashboard(db_manager, clusterer):
    """Display job dashboard with visualizations"""
    st.title("📊 Job Dashboard")
    
    try:
        jobs_df = db_manager.get_jobs()
        
        if jobs_df.empty:
            st.warning("No jobs found. Please scrape some jobs first!")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_locations = st.multiselect(
                "Filter by Location",
                options=jobs_df['Location'].unique(),
                default=jobs_df['Location'].unique()[:5] if len(jobs_df['Location'].unique()) > 5 else jobs_df['Location'].unique()
            )
        
        with col2:
            selected_companies = st.multiselect(
                "Filter by Company",
                options=jobs_df['Company'].unique(),
                default=jobs_df['Company'].unique()[:5] if len(jobs_df['Company'].unique()) > 5 else jobs_df['Company'].unique()
            )
        
        # Filter data
        if selected_locations:
            jobs_df = jobs_df[jobs_df['Location'].isin(selected_locations)]
        if selected_companies:
            jobs_df = jobs_df[jobs_df['Company'].isin(selected_companies)]
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Jobs by location
            location_counts = jobs_df['Location'].value_counts().head(10)
            fig = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Jobs by Location (Top 10)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Jobs by company
            company_counts = jobs_df['Company'].value_counts().head(10)
            fig = px.bar(
                x=company_counts.values,
                y=company_counts.index,
                orientation='h',
                title="Jobs by Company (Top 10)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster visualization
        if 'cluster_id' in jobs_df.columns:
            st.subheader("Job Clusters")
            cluster_counts = jobs_df['cluster_id'].value_counts().sort_index()
            
            fig = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Distribution of Jobs by Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent jobs table
        st.subheader("Recent Jobs")
        display_cols = ['Title', 'Company', 'Location', 'Skills']
        if 'cluster_id' in jobs_df.columns:
            display_cols.append('cluster_id')
        
        st.dataframe(
            jobs_df[display_cols].head(20),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

def show_skill_matching(db_manager, clusterer, notification_manager):
    """Display skill matching interface"""
    st.title("🎯 Skill Matching")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Skills")
        
        # User input form
        with st.form("skill_form"):
            user_skills = st.text_area(
                "Enter your skills (comma-separated)",
                placeholder="Python, Machine Learning, Data Analysis, SQL, Pandas",
                height=100
            )
            
            email = st.text_input(
                "Email (for notifications)",
                placeholder="your.email@example.com"
            )
            
            submit_button = st.form_submit_button("Find Matching Jobs")
        
        if submit_button and user_skills:
            # Save user preferences
            if email:
                notification_manager.add_user_preference(email, user_skills)
                st.success("Preferences saved!")
            
            # Find matching jobs
            try:
                jobs_df = db_manager.get_jobs()
                
                if jobs_df.empty:
                    st.warning("No jobs in database!")
                    return
                
                # Simple skill matching (can be enhanced)
                user_skill_list = [skill.strip().lower() for skill in user_skills.split(',')]
                
                def calculate_match_score(job_skills):
                    if pd.isna(job_skills) or job_skills == "":
                        return 0
                    
                    job_skill_list = [skill.strip().lower() for skill in str(job_skills).split(',')]
                    matches = sum(1 for skill in user_skill_list 
                                if any(skill in job_skill or job_skill in skill 
                                      for job_skill in job_skill_list))
                    return matches / len(user_skill_list) * 100
                
                jobs_df['match_score'] = jobs_df['Skills'].apply(calculate_match_score)
                matching_jobs = jobs_df[jobs_df['match_score'] > 0].sort_values(
                    'match_score', ascending=False
                )
                
                st.session_state['matching_jobs'] = matching_jobs
                
            except Exception as e:
                st.error(f"Error finding matches: {e}")
    
    with col2:
        st.subheader("Matching Jobs")
        
        if 'matching_jobs' in st.session_state:
            matching_jobs = st.session_state['matching_jobs']
            
            if not matching_jobs.empty:
                st.success(f"Found {len(matching_jobs)} matching jobs!")
                
                # Display matching jobs
                for idx, (_, job) in enumerate(matching_jobs.head(10).iterrows()):
                    with st.expander(f"🎯 {job['Title']} at {job['Company']} (Match: {job['match_score']:.1f}%)"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Location:** {job['Location']}")
                            st.write(f"**Experience:** {job['Experience']}")
                        
                        with col_b:
                            st.write(f"**Skills:** {job['Skills']}")
                            if 'cluster_id' in job:
                                st.write(f"**Cluster:** {job['cluster_id']}")
                        
                        if job['Summary']:
                            st.write(f"**Summary:** {job['Summary'][:200]}...")
            else:
                st.warning("No matching jobs found. Try different skills or check if jobs are available in the database.")
        else:
            st.info("Enter your skills to find matching jobs!")

def show_admin_panel(scraper, db_manager, clusterer):
    """Display admin panel for manual operations"""
    st.title("⚙️ Admin Panel")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Scrape Jobs", "🤖 Train Model", "🗄️ Database Management"])
    
    with tab1:
        st.subheader("Job Scraping")
        
        col1, col2 = st.columns(2)
        with col1:
            keyword = st.text_input("Search Keyword", value="data science")
            pages = st.number_input("Pages to Scrape", min_value=1, max_value=10, value=2)
        
        with col2:
            st.info(f"Will scrape approximately {pages * 20} jobs")
        
        if st.button("🚀 Start Scraping", type="primary"):
            with st.spinner("Scraping jobs..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Scrape jobs
                    jobs_df = scraper.scrape_jobs(keyword=keyword, pages=pages)
                    progress_bar.progress(50)
                    status_text.text("Scraping completed. Saving to database...")
                    
                    if not jobs_df.empty:
                        # Save to database
                        db_manager.save_jobs(jobs_df)
                        progress_bar.progress(100)
                        status_text.text("Jobs saved successfully!")
                        
                        st.success(f"Successfully scraped and saved {len(jobs_df)} jobs!")
                        st.dataframe(jobs_df.head())
                    else:
                        st.warning("No jobs were scraped. Please check the keyword or try again.")
                        
                except Exception as e:
                    st.error(f"Error during scraping: {e}")
    
    with tab2:
        st.subheader("Machine Learning Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            clustering_method = st.selectbox(
                "Clustering Algorithm",
                ["kmeans", "dbscan"]
            )
        
        with col2:
            if clustering_method == "kmeans":
                max_clusters = st.number_input("Max Clusters to Test", min_value=2, max_value=15, value=8)
        
        if st.button("🤖 Train Clustering Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Get jobs from database
                    jobs_df = db_manager.get_jobs()
                    
                    if jobs_df.empty:
                        st.warning("No jobs in database to train on!")
                        return
                    
                    # Train clustering model
                    clustered_jobs, cluster_summary = clusterer.cluster_jobs(
                        jobs_df, method=clustering_method
                    )
                    
                    # Update database with cluster assignments
                    conn = sqlite3.connect(DATABASE_FILE)
                    for idx, row in clustered_jobs.iterrows():
                        conn.execute(
                            "UPDATE jobs SET cluster_id = ? WHERE id = ?",
                            (int(row['cluster_id']), int(row.name + 1))
                        )
                    conn.commit()
                    conn.close()
                    
                    # Save model
                    clusterer.save_model()
                    
                    st.success(f"Model trained successfully with {clusterer.n_clusters} clusters!")
                    
                    # Display cluster summary
                    if cluster_summary:
                        st.subheader("Cluster Summary")
                        for cluster_id, summary in cluster_summary.items():
                            with st.expander(f"Cluster {cluster_id} ({summary['size']} jobs)"):
                                st.write(f"**Common Skills:** {', '.join(summary['common_skills'][:5])}")
                                st.write(f"**Common Titles:** {', '.join(summary['common_titles'][:3])}")
                    
                except Exception as e:
                    st.error(f"Error training model: {e}")
    
    with tab3:
        st.subheader("Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Show Database Stats"):
                try:
                    jobs_df = db_manager.get_jobs()
                    st.write(f"**Total Jobs:** {len(jobs_df)}")
                    st.write(f"**Unique Companies:** {jobs_df['Company'].nunique()}")
                    st.write(f"**Unique Locations:** {jobs_df['Location'].nunique()}")
                    if 'cluster_id' in jobs_df.columns:
                        st.write(f"**Clusters:** {jobs_df['cluster_id'].nunique()}")
                except Exception as e:
                    st.error(f"Error getting stats: {e}")
        
        with col2:
            days_to_keep = st.number_input("Days of jobs to keep", min_value=1, value=30)
            if st.button("🗑️ Clean Old Jobs", type="secondary"):
                try:
                    db_manager.clear_old_jobs(days_to_keep)
                    st.success(f"Cleaned jobs older than {days_to_keep} days")
                except Exception as e:
                    st.error(f"Error cleaning jobs: {e}")
        
        # Download data
        st.subheader("Export Data")
        try:
            jobs_df = db_manager.get_jobs()
            if not jobs_df.empty:
                csv = jobs_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Jobs as CSV",
                    data=csv,
                    file_name=f"jobs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error preparing download: {e}")

def show_notifications_page(notification_manager):
    """Display notifications configuration page"""
    st.title("📧 Notifications")
    
    tab1, tab2 = st.tabs(["⚙️ Email Setup", "👥 User Preferences"])
    
    with tab1:
        st.subheader("Email Configuration")
        st.info("Configure email settings to send job notifications")
        
        with st.form("email_config"):
            smtp_server = st.text_input("SMTP Server", placeholder="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            email_user = st.text_input("Email Username", placeholder="your.email@gmail.com")
            email_pass = st.text_input("Email Password", type="password", 
                                     help="Use app password for Gmail")
            
            test_email = st.text_input("Test Email", placeholder="test@example.com")
            
            col1, col2 = st.columns(2)
            with col1:
                save_config = st.form_submit_button("💾 Save Configuration")
            with col2:
                test_email_btn = st.form_submit_button("📧 Send Test Email")
        
        if save_config:
            # Save email configuration (in production, use secure storage)
            st.session_state['email_config'] = {
                'smtp_server': smtp_server,
                'smtp_port': smtp_port,
                'email_user': email_user,
                'email_pass': email_pass
            }
            st.success("Email configuration saved!")
        
        if test_email_btn and test_email:
            if 'email_config' in st.session_state:
                config = st.session_state['email_config']
                # Create a dummy job for testing
                test_jobs = pd.DataFrame([{
                    'Title': 'Test Job Notification',
                    'Company': 'Test Company',
                    'Location': 'Test Location',
                    'Skills': 'Test Skills'
                }])
                
                success = notification_manager.send_email_notification(
                    test_email, test_jobs, **config
                )
                
                if success:
                    st.success("Test email sent successfully!")
                else:
                    st.error("Failed to send test email. Check configuration.")
            else:
                st.warning("Please save email configuration first!")
    
    with tab2:
        st.subheader("User Preferences")
        
        # Display current user preferences
        if notification_manager.user_preferences:
            st.write("**Current Users:**")
            for email, prefs in notification_manager.user_preferences.items():
                with st.expander(f"📧 {email}"):
                    st.write(f"**Skills:** {prefs['skills']}")
                    st.write(f"**Registered:** {prefs['created_at']}")
                    
                    if st.button(f"🗑️ Remove {email}", key=f"remove_{email}"):
                        del notification_manager.user_preferences[email]
                        notification_manager.save_user_preferences()
                        st.success(f"Removed {email}")
                        st.rerun()
        else:
            st.info("No user preferences registered yet.")
        
        # Manual notification sending
        st.subheader("Send Manual Notifications")
        if st.button("📤 Send Notifications to All Users"):
            if 'email_config' not in st.session_state:
                st.warning("Please configure email settings first!")
                return
            
            try:
                db_manager = DatabaseManager()
                jobs_df = db_manager.get_jobs(limit=50)  # Get recent jobs
                
                if jobs_df.empty:
                    st.warning("No jobs to notify about!")
                    return
                
                config = st.session_state['email_config']
                sent_count = 0
                
                for email, prefs in notification_manager.user_preferences.items():
                    # Simple matching logic (can be enhanced)
                    user_skills = [s.strip().lower() for s in prefs['skills'].split(',')]
                    
                    matching_jobs = jobs_df[
                        jobs_df['Skills'].str.lower().str.contains(
                            '|'.join(user_skills), na=False
                        )
                    ]
                    
                    if not matching_jobs.empty:
                        success = notification_manager.send_email_notification(
                            email, matching_jobs.head(5), **config
                        )
                        if success:
                            sent_count += 1
                
                st.success(f"Sent notifications to {sent_count} users!")
                
            except Exception as e:
                st.error(f"Error sending notifications: {e}")

# Automation functions (for background scheduling)
def automated_job_scraping():
    """Automated job scraping function (can be scheduled)"""
    try:
        scraper = JobScraper()
        db_manager = DatabaseManager()
        clusterer = JobClusterer()
        
        # Scrape jobs with different keywords
        keywords = ["data science", "python developer", "machine learning", "data analyst"]
        
        for keyword in keywords:
            jobs_df = scraper.scrape_jobs(keyword=keyword, pages=1)
            if not jobs_df.empty:
                db_manager.save_jobs(jobs_df)
        
        # Retrain model if we have enough new data
        all_jobs = db_manager.get_jobs()
        if len(all_jobs) > 50:
            clustered_jobs, _ = clusterer.cluster_jobs(all_jobs)
            clusterer.save_model()
            
            # Update database with new clusters
            conn = sqlite3.connect(DATABASE_FILE)
            for idx, row in clustered_jobs.iterrows():
                conn.execute(
                    "UPDATE jobs SET cluster_id = ? WHERE id = ?",
                    (int(row['cluster_id']), int(row.name + 1))
                )
            conn.commit()
            conn.close()
        
        print(f"Automated scraping completed at {datetime.now()}")
        
    except Exception as e:
        print(f"Error in automated scraping: {e}")

def schedule_automation():
    """Set up automated scheduling"""
    # Schedule daily job scraping
    schedule.every().day.at("09:00").do(automated_job_scraping)
    
    # Run scheduler in background
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

# Additional utility functions
def create_sample_data():
    """Create sample data for testing"""
    sample_jobs = pd.DataFrame([
        {
            'Title': 'Data Scientist',
            'Company': 'Tech Corp',
            'Location': 'Bangalore',
            'Experience': '2-5 years',
            'Summary': 'Looking for a data scientist with Python and ML experience',
            'Skills': 'Python, Machine Learning, Pandas, Scikit-learn',
            'Combined_Text': 'data scientist python machine learning pandas scikit-learn',
            'Post_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Keyword': 'data science'
        },
        {
            'Title': 'Python Developer',
            'Company': 'Software Inc',
            'Location': 'Mumbai',
            'Experience': '1-3 years',
            'Summary': 'Python developer for web applications',
            'Skills': 'Python, Django, Flask, PostgreSQL',
            'Combined_Text': 'python developer django flask postgresql web applications',
            'Post_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Keyword': 'python developer'
        },
        {
            'Title': 'Machine Learning Engineer',
            'Company': 'AI Solutions',
            'Location': 'Hyderabad',
            'Experience': '3-6 years',
            'Summary': 'ML engineer for production systems',
            'Skills': 'Python, TensorFlow, PyTorch, AWS, Docker',
            'Combined_Text': 'machine learning engineer python tensorflow pytorch aws docker',
            'Post_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Keyword': 'machine learning'
        }
    ])
    return sample_jobs

# Sidebar additional features
def add_sidebar_features():
    """Add additional features to sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Quick Actions")
    
    if st.sidebar.button("📊 Create Sample Data"):
        try:
            db_manager = DatabaseManager()
            sample_jobs = create_sample_data()
            db_manager.save_jobs(sample_jobs)
            st.sidebar.success("Sample data created!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    if st.sidebar.button("🔄 Start Automation"):
        try:
            schedule_automation()
            st.sidebar.success("Automation started!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # App statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 App Stats")
    try:
        db_manager = DatabaseManager()
        jobs_df = db_manager.get_jobs()
        
        if not jobs_df.empty:
            st.sidebar.metric("Total Jobs", len(jobs_df))
            st.sidebar.metric("Companies", jobs_df['Company'].nunique())
            
            # Latest job info
            latest_job = jobs_df.iloc[0] if len(jobs_df) > 0 else None
            if latest_job is not None:
                st.sidebar.write("**Latest Job:**")
                st.sidebar.write(f"*{latest_job['Title']}*")
                st.sidebar.write(f"at {latest_job['Company']}")
        else:
            st.sidebar.info("No jobs yet")
    except:
        pass

if __name__ == "__main__":
    # Add sidebar features
    add_sidebar_features()
    
    # Run main application
    main()
