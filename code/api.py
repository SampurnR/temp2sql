import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
# from google import genai  # Gemini (Google GenAI)
# from openai import AzureOpenAI  # Azure OpenAI (direct SDK)
from langchain_openai import AzureChatOpenAI  # Azure OpenAI (LangChain)
from pydantic import BaseModel

from helper import setup_database, text2sql

load_dotenv()

# genai_client = None  # Gemini (Google GenAI)
aioai_client = None  # Azure OpenAI


@asynccontextmanager
async def lifespan(app: FastAPI):
    global aioai_client
    # --- Gemini (Google GenAI) ---
    # global genai_client
    # genai_client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    # --- Azure OpenAI (direct SDK) ---
    # aioai_client = AzureOpenAI(
    #     api_key=os.environ['AZURE_OPENAI_API_KEY'],
    #     api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    #     azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
    # )

    # --- Azure OpenAI (LangChain) ---
    aioai_client = AzureChatOpenAI(
        azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        api_version=os.environ['AZURE_OPENAI_API_VERSION']
    )

    if not os.path.exists('ecommerce.db'):
        setup_database(db_name='ecommerce.db', data_dir='../data')

    yield


app = FastAPI(title="Text-to-SQL API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    sql: str
    results: list[dict]
    row_count: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # sql, results_df = text2sql(genai_client, request.query)  # Gemini (Google GenAI)
    sql, results_df = text2sql(aioai_client, request.query)  # Azure OpenAI
    if results_df is not None:
        results = results_df.to_dict(orient='records')
        return QueryResponse(sql=sql, results=results, row_count=len(results))
    return QueryResponse(sql=sql, results=[], row_count=0)
