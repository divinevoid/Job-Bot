# Job-Bot
Job-Bot is an AI-powered tool that utilizes vector databases for job postings and AI-generated responses. This project is designed to assist with job search tasks by analyzing job data and providing relevant information using a conversational agent.
## Setup Instructions

Follow the steps below to set up and run the project.

### 1. Clone the Repository
Clone the repository using the following command:
```bash
git clone https://github.com/divinevoid/Job-Bot.git
```
### 2. **Install all the dependecies
````bash
cd Job-Bot
pip install -r requirements.txt
````
### 3. Ensure the api keys
Ensure you have all the required api keys. Make sure to set your api keys in the environmental variables or in a .env file.
### 4. Make connection to Google Cloud
   To make connection run the following:
```bash
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project <your_project_id>
```
### 5. Build Vector Store
````bash
python vector.py
````
### 6. Run the Job Agent Streamlit App
````bash
streamlit run job_agent.py
````
