# Job-Bot
To run the code we need to follow the step:
1. **Clone the repository**
   To clone the repository run
   git clone https://github.com/divinevoid/Job-Bot.git
2. **Install all the dependecies**
   run "pip install -r requirements.txt"
3. **Ensure the api keys**: Ensure you have all the required api keys to establish connection
4. **Make connection to Google Cloud**
   To make connection run the following:
   "gcloud auth login"
   "gcloud auth application-default login"
   "gcloud auth application-default set-quota-project <your_project_id>"
5. Run vector.py using "python vector.py" to build chromadb vectorstore.
6. Now you can run the Job Agent Streamlit app using "streamlit run job_agent.py"
