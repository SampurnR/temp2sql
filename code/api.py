import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import AzureChatOpenAI  # Azure OpenAI (LangChain)
from pydantic import BaseModel

from helper import setup_database, text2sql

load_dotenv()

aioai_client = AzureChatOpenAI(
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION']
)

if not os.path.exists('ecommerce.db'):
    setup_database(db_name='ecommerce.db', data_dir='../data')

app = FastAPI(title="Text-to-SQL API")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    sql: str
    results: list[dict]
    row_count: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/text2sql", response_model=QueryResponse)
def query(request: QueryRequest):
    # sql, results_df = text2sql(genai_client, request.query)  # Gemini (Google GenAI)
    sql, results_df = text2sql(aioai_client, request.query)  # Azure OpenAI
    if results_df is not None:
        results = results_df.to_dict(orient='records')
        return QueryResponse(sql=sql, results=results, row_count=len(results))
    return QueryResponse(sql=sql, results=[], row_count=0)
