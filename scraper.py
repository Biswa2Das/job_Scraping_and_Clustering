import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company_tag = job.find("a", href=lambda x: x and "Employer-Profile" in x)
                company = company_tag.get_text(strip=True) if company_tag else ""
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                skills_tag = job.find("span", string="Key Skills")
                skills = skills_tag.find_next("p").get_text(strip=True) if skills_tag else ""
                job_link_tag = job.find("a", href=True)
                link = "https://www.karkidi.com" + job_link_tag["href"] if job_link_tag else ""

                jobs_list.append({
                    "title": title,
                    "company": company,
                    "location": location,
                    "experience": experience,
                    "skills": skills,
                    "link": link
                })
            except Exception as e:
                continue

        time.sleep(1)

    return pd.DataFrame(jobs_list)
