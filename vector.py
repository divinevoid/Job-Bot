from langchain.vectorstores import Chroma
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.cloud import bigquery
from dotenv import load_dotenv
from storage_config import PROJECT_ID, DATASET_ID, JOB_POSTING_TABLE_NAME

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

client = bigquery.Client(
    # credentials=AnonymousCredentials,
    project="duallens",
    _http=bigquery.Client()._http,
)
client._http.headers["Authorization"] = f"Bearer {api_key}"
table_id = f"{PROJECT_ID}.{DATASET_ID}.{JOB_POSTING_TABLE_NAME}"

query = f"SELECT * FROM `{table_id}`"
query_job = client.query(query)
results = query_job.result()
rows = [dict(row) for row in results]

documents = []
for row in rows:
    content = f"Job ID: {row['Job ID']}, Position: {row['Position']}, Location: {row['Location']}, Key Responsibilities: {row['Key Responsibilities']}, Qualifications: {row['Qualifications']}"
    documents.append({"content": content})

documents[0]

texts = [doc["content"] for doc in documents]
metadatas = [
    {"Job ID": row["Job ID"], "Position": row["Position"], "Location": row["Location"]}
    for row in rows
]

vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory="chroma_db_index_job",
)

vector_store.persist()
