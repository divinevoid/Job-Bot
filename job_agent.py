from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.cloud import bigquery, storage
from google.auth.credentials import AnonymousCredentials
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from dotenv import load_dotenv
from storage_config import (
    PROJECT_ID,
    DATASET_ID,
    JOB_APPLICANTS_TABLE_NAME,
    JOB_POSTING_TABLE_NAME,
)
from langchain.agents import initialize_agent, Tool
import streamlit as st
from datetime import datetime
import tempfile
import os
import pysqlite3 as sqlite3

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")
cred = AnonymousCredentials()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

storage_client = storage.Client(
    credentials=cred,
    project="duallens",
    _http=storage.Client()._http,
)

client = bigquery.Client(
    credentials=cred,
    project="duallens",
    _http=bigquery.Client()._http,
)
client._http.headers["Authorization"] = f"Bearer {api_key}"

table_id = f"{PROJECT_ID}.{DATASET_ID}."

vector_store = Chroma(
    persist_directory="chroma_db_index_job", embedding_function=embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def save_resume_to_storage(resume_file, job_id):
    try:
        filename = f"{job_id}_{resume_file.name}"
        BUCKET_NAME = "resume_bucket_1"
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.upload_from_string(resume_file.read(), content_type=resume_file.type)
        resume_url = blob.public_url
        st.session_state.resume_url = resume_url
        st.success(f"Resume uploaded successfully: {resume_url}")
        return resume_url
    except Exception as e:
        st.error(f"Failed to upload resume: {str(e)}")
        return None


def save_to_bq(data):
    try:
        errors = client.insert_rows_json(
            f"{table_id}.{JOB_APPLICANTS_TABLE_NAME}", [data]
        )
        if errors:
            st.error(f"Error inserting rows: {errors}")
        else:
            st.success("Application saved!")
    except Exception as e:
        st.error(f"Failed to submit the application: {str(e)}")


# def process_data(data, criteria=None):
#     """
#     Args:
#        data(list): The input data to process
#        criteria(fucntion, optional): A callable function to filter data
#     Returns:
#        list: Processed Data
#     """
#     if criteria:
#         return [item for item in data if criteria(item)]
#     return [item for item in data if item]


# Function to query BigQuery tool
def query_bigquery_tool(input_text: str):
    query = f"""
    SELECT *
    FROM `{table_id}.{JOB_POSTING_TABLE_NAME}`
    WHERE POSITION LIKE @position
    OR LOCATION LIKE @location
    """
    query_job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("position", "STRING", f"%{input_text}%"),
                bigquery.ScalarQueryParameter("location", "STRING", f"%{input_text}%"),
            ]
        ),
    )
    raw_results = [dict(row) for row in query_job.result()]
    filtered_results = [job for job in raw_results if job]

    # Return the filtered results
    return filtered_results

    # def criteria(item):
    #     return any(
    #         key in item and item[key] for key in ["Job ID", "Position", "Location"]
    #     )

    # cleaned_results = process_data(raw_results, criteria)
    # return raw_results


# function to retrieve from vectorstore tool
def vector_store_tool(input_text: str):
    docs = retriever.get_relevant_documents(input_text)
    raw_results = [doc.metadata for doc in docs]
    filtered_results = [job for job in raw_results if job]

    # Return the filtered results
    return filtered_results

    # def criteria(item):
    #     return any(
    #         key in item and item[key] for key in ["Job ID", "Position", "Location"]
    #     )

    # cleaned_results = process_data(raw_results, criteria)
    # return raw_results


tools = [
    Tool(
        name="BigQueryTool",
        func=query_bigquery_tool,
        description=(
            """Use this tool to retrieve job postings from the BigQuery database based on user input and the context of the user input. Provide job(s) strictly based on user context.
               Always use this tool first before going to vectorstore. If you find any relevant job posting respond with them in a json format:
            I have found the following job:
            [{{"Job ID": __, "Position": __, "Location": __}}]
            If you find multiple positions respond with:
            I have found the following jobs:
            [{{"Job ID": __, "Position": __, "Location": __}},
            {{"Job ID": __, "Position": __, "Location": __}}]
            .... and so on.
            - If a job has additional context or is potentially related to another role, include an optional `"Note"` field with a brief description of the context.
            [{{"Job ID": __, "Position": __, "Location": __}},
            {{"Job ID": __, "Position": __, "Location": __, "Note": __}}]
            .... and so on.
            Provide me with a clean formatting for the response.


            """
        ),
    ),
    Tool(
        name="VectorStoreTool",
        func=vector_store_tool,
        description=(
            """Use this tool to retrieve job postings from the vectorstore based on user input and the context of the user input. Provide job(s) strictly based on user context.
            You are a helpful assistant for job search, filter the below given job data based on user query. 
            If you find any relevant job posting respond with them in a json format:
            I have found the following job:
            [{{"Job ID": __, "Position": __, "Location": __}}]
            If you find multiple positions respond with:
            I have found the following jobs:
            [{{"Job ID": __, "Position": __, "Location": __}},
            {{"Job ID": __, "Position": __, "Location": __}}]
            .... and so on.
            - If a job has additional context or is potentially related to another role, include an optional `"Note"` field with a brief description of the context.
            [{{"Job ID": __, "Position": __, "Location": __}},
            {{"Job ID": __, "Position": __, "Location": __, "Note": __}}]
            .... and so on.
            Filter the retrieved metadata job result that matches user asked query criteria.
            Provide me with a clean formatting for the response.
            """
        ),
    ),
]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
)

# agent.run("Machine learning freshers job")

# Streamlit App
st.title("Job Application Assistant")

# Initialize session state
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False
    st.session_state.resume_url = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_results" not in st.session_state:
    st.session_state.job_results = []
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "application_complete" not in st.session_state:
    st.session_state.application_complete = False

# Upload Resume Section
st.sidebar.header("INSTRUCTIONS")
st.sidebar.write("1. You can search for jobs.")
st.sidebar.write("2. You can upload your resume to find personalised jobs.")
uploaded_file = st.sidebar.file_uploader("Upload your resume (PDF only):", type=["pdf"])

if uploaded_file and not st.session_state.resume_uploaded:
    # job_id = "temp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    resume_loader = PyPDFLoader(temp_file_path)
    pages = resume_loader.load()
    resume_text = " ".join([page.page_content for page in pages])
    # print(resume_text)
    st.session_state.resume_uploaded = True
    st.session_state.resume_text = resume_text
    st.success("Resume processed and stored!")


# User Query Section
user_input = st.text_input("Ask me about jobs or provide a query:")
if user_input:
    # Combine resume text if available
    combined_input = (
        f"Based on this resume: {st.session_state.resume_text}. {user_input}"
        if st.session_state.resume_text
        else user_input
    )
    response = agent.run(combined_input)
    # Parse and display job results
    if "I have found the following jobs:" in response:
        json_str = response.split("I have found the following jobs:")[1].strip()
        try:
            st.session_state.job_results = json.loads(json_str)
        except json.JSONDecodeError:
            st.write("Error parsing job results. Please refine your query.")
    else:
        st.write(response)

# Job Results Section
if st.session_state.job_results and not st.session_state.selected_job:
    st.subheader("Job Results")
    for job in st.session_state.job_results:
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            if st.button(f"Job ID {job['Job ID']}"):
                st.session_state.selected_job = job
        with col2:
            st.write(f"**Position**: {job['Position']}")
        with col3:
            st.write(f"**Location**: {job['Location']}")


if st.session_state.selected_job and not st.session_state.application_complete:
    st.subheader("Job Application")
    selected_job = st.session_state.selected_job
    st.write(f"**Job ID**: {selected_job['Job ID']}")
    st.write(f"**Position**: {selected_job['Position']}")
    st.write(f"**Location**: {selected_job['Location']}")

    name = st.text_input("Enter your name:")
    email = st.text_input("Enter your email:")
    mobile = st.text_input("Enter your mobile number:")

    if not st.session_state.resume_uploaded:
        uploaded_resume = st.file_uploader(
            "Upload your resume (PDF only):", type=["pdf"], key="resume_uploader_2"
        )
        if uploaded_resume:
            job_id = selected_job["Job ID"]
            resume_url = save_resume_to_storage(uploaded_resume, job_id)
    else:
        resume_url = st.session_state.resume_url

    if st.button("Submit Application"):
        if name and email and mobile and resume_url:
            application_data = {
                "job_id": selected_job["Job ID"],
                "user_name": name,
                "email": email,
                "resume_url": resume_url,
                "application_date": datetime.now().isoformat(),
                "mobile": mobile,
            }
            save_to_bq(application_data)
            st.session_state.application_complete = True
        else:
            st.error("Please complete all fields before submitting.")

if st.session_state.application_complete:
    st.success("Your application has been submitted successfully!")

if st.session_state.resume_uploaded:
    st.subheader("Your Resume Details")
    st.write("**Resume URL**: ", st.session_state.resume_url)
    st.text_area("Resume Content:", st.session_state.resume_text, height=300)
    if st.button("Remove Resume"):
        st.session_state.resume_uploaded = False
        st.session_state.resume_url = None
        st.session_state.resume_text = ""
        st.success("Resume removed successfully!")
