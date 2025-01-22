import os
from dotenv import load_dotenv


load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("API_KEY")


PROJECT_ID = "duallens"
DATASET_ID = "Duallens"
BLOG_TABLE_NAME = "blogs"
JOB_POSTING_TABLE_NAME = "job_postings"
JOB_APPLICANTS_TABLE_NAME = "applicants"
