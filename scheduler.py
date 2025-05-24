# scheduler.py
"""
Daily job scraping scheduler script
Run this script via cron job or task scheduler for automated daily scraping
"""

import os
import sys
import schedule
import time
import logging
from datetime import datetime
import pandas as pd
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add the main directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our classes (assuming they're in the same directory)
try:
    from streamlit_app import KarkidiJobScraper, JobClusteringModel, JobAlertSystem
except ImportError:
    print("Error: Could not import required classes. Make sure streamlit_app.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('job_scraper.log'),
        logging.StreamHandler()
    ]
)

class JobScrapingScheduler:
    def __init__(self, config_file='scheduler_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.scraper = KarkidiJobScraper()
        self.model = JobClusteringModel()
        self.alert_system = JobAlertSystem()
        
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            "keywords": ["data science", "python developer", "machine learning", "software engineer"],
            "pages_per_keyword": 2,
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "enabled": False
            },
            "auto_retrain_model": True,
            "min_jobs_for_retrain": 50
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logging.error(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logging.info(f"Created default config file: {self.config_file}")
            return default_config
    
    def daily_scraping_job(self):
        """Main job that runs daily"""
        logging.info("Starting daily job scraping...")
        
        try:
            # Load existing data
            existing_jobs = []
            if os.path.exists("scraped_jobs.csv"):
                df_existing = pd.read_csv("scraped_jobs.csv")
                existing_jobs = len(df_existing)
                logging.info(f"Found {existing_jobs} existing jobs")
            
            # Scrape new jobs
            all_new_jobs = []
            
            for keyword in self.config['keywords']:
                logging.info(f"Scraping jobs for keyword: {keyword}")
                try:
                    df_new = self.scraper.scrape_jobs(
                        keyword=keyword,
                        pages=self.config['pages_per_keyword']
                    )
                    
                    if not df_new.empty:
                        all_new_jobs.append(df_new)
                        logging.info(f"Scraped {len(df_new)} jobs for '{keyword}'")
                    else:
                        logging.warning(f"No jobs found for keyword: {keyword}")
                        
                except Exception as e:
                    logging.error(f"Error scraping jobs for '{keyword}': {e}")
                    continue
                
                # Add delay between keywords
                time.sleep(5)
            
            # Combine all new jobs
            if all_new_jobs:
                df_all_new = pd.concat(all_new_jobs, ignore_index=True)
                
                # Remove duplicates within new data
                df_all_new.drop_duplicates(subset=['Title', 'Company'], keep='first', inplace=True)
                
                # Combine with existing data
                if os.path.exists("scraped_jobs.csv"):
                    df_existing = pd.read_csv("scraped_jobs.csv")
                    df_combined = pd.concat([df_existing, df_all_new], ignore_index=True)
                    
                    # Remove duplicates between old and new
                    df_combined.drop_duplicates(subset=['Title', 'Company'], keep='last', inplace=True)
                else:
                    df_combined = df_all_new
                
                # Save updated data
                df_combined.to_csv("scraped_jobs.csv", index=False)
                
                new_jobs_count = len(df_all_new)
                total_jobs = len(df_combined)
                
                logging.info(f"Scraping completed: {new_jobs_count} new jobs, {total_jobs} total jobs")
                
                # Retrain model if needed
                if self.config['auto_retrain_model'] and new_jobs_count >= self.config['min_jobs_for_retrain']:
                    logging.info("Retraining clustering model...")
                    try:
                        self.model.train_model(df_combined)
                        logging.info("Model retrained successfully")
                    except Exception as e:
                        logging.error(f"Error retraining model: {e}")
                
                # Check for alerts
                self.check_and_send_alerts(df_all_new)
                
            else:
                logging.warning("No new jobs scraped from any keyword")
                
        except Exception as e:
            logging.error(f"Error in daily scraping job: {e}")
    
    def check_and_send_alerts(self, new_jobs_df):
        """Check for job matches and send alerts"""
        if not self.config['email']['enabled']:
            logging.info("Email alerts disabled")
            return
        
        try:
            # Load the model
            if not self.model.load_model():
                logging.warning("No trained model found for alerts")
                return
            
            # Check for matches
            matches = self.alert_system.check_new_matches(new_jobs_df, self.model)
            
            if matches:
                logging.info(f"Found matches for {len(matches)} users")
                
                for email, matched_jobs in matches.items():
                    try:
                        self.send_alert_email(email, matched_jobs)
                        logging.info(f"Alert sent to {email} for {len(matched_jobs)} jobs")
                    except Exception as e:
                        logging.error(f"Error sending alert to {email}: {e}")
            else:
                logging.info("No job matches found for alerts")
                
        except Exception as e:
            logging.error(f"Error checking alerts: {e}")
    
    def send_alert_email(self, recipient_email, matched_jobs):
        """Send email alert for matched jobs"""
        if not self.config['email']['sender_email'] or not self.config['email']['sender_password']:
            logging.error("Email credentials not configured")
            return
        
        # Create email content
        subject = f"🔔 {len(matched_jobs)} New Job Matches Found!"
        
        html_content = f"""
        <html>
        <body>
            <h2>New Job Matches Found!</h2>
            <p>Hi! We found {len(matched_jobs)} new job listings that match your preferred skill clusters.</p>
            
            <h3>Job Listings:</h3>
        """
        
        for i, job in enumerate(matched_jobs[:10], 1):  # Limit to 10 jobs per email
            html_content += f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0;">
                <h4>{job.get('Title', 'N/A')}</h4>
                <p><strong>Company:</strong> {job.get('Company', 'N/A')}</p>
                <p><strong>Location:</strong> {job.get('Location', 'N/A')}</p>
                <p><strong>Experience:</strong> {job.get('Experience', 'N/A')}</p>
                <p><strong>Skills:</strong> {job.get('Skills', 'N/A')[:200]}...</p>
            </div>
            """
        
        if len(matched_jobs) > 10:
            html_content += f"<p><em>... and {len(matched_jobs) - 10} more jobs!</em></p>"
        
        html_content += """
            <hr>
            <p><small>This is an automated alert from the Karkidi Job Scraper. 
            To modify your preferences, please use the Streamlit app.</small></p>
        </body>
        </html>
        """
        
        # Send email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.config['email']['sender_email']
        msg['To'] = recipient_email
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        with smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port']) as server:
            server.starttls()
            server.login(self.config['email']['sender_email'], self.config['email']['sender_password'])
            server.send_message(msg)
    
    def run_scheduler(self):
        """Run the scheduler"""
        logging.info("Starting job scraping scheduler...")
        
        # Schedule daily job at 9 AM
        schedule.every().day.at("09:00").do(self.daily_scraping_job)
        
        # Optional: Schedule additional runs
        # schedule.every().day.at("18:00").do(self.daily_scraping_job)
        
        logging.info("Scheduler configured. Waiting for scheduled times...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_once(self):
        """Run scraping job once (for testing)"""
        logging.info("Running scraping job once...")
        self.daily_scraping_job()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Karkidi Job Scraping Scheduler')
    parser.add_argument('--run-once', action='store_true', help='Run scraping job once and exit')
    parser.add_argument('--config', default='scheduler_config.json', help='Config file path')
    
    args = parser.parse_args()
    
    scheduler = JobScrapingScheduler(args.config)
    
    if args.run_once:
        scheduler.run_once()
    else:
        scheduler.run_scheduler()

if __name__ == "__main__":
    main()
