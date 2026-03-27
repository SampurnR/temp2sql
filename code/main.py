import os
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
# from google import genai  # Gemini (Google GenAI)
# from openai import AzureOpenAI  # Azure OpenAI (direct SDK)
from langchain_openai import AzureChatOpenAI  # Azure OpenAI (LangChain)

from helper import DB_PATH, setup_database, text2sql

load_dotenv()

# --- Gemini (Google GenAI) ---
# genai_client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

# --- Azure OpenAI (direct SDK) ---
# genai_client = AzureOpenAI(
#     api_key=os.environ['AZURE_OPENAI_API_KEY'],
#     api_version=os.environ['AZURE_OPENAI_API_VERSION'],
#     azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
# )

# --- Azure OpenAI (LangChain) ---
genai_client = AzureChatOpenAI(
    azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION']
)


def gradio_text2sql(user_query):
    """Wrapper for Gradio: returns both the SQL and the results table."""
    sql, results_df = text2sql(genai_client, user_query)
    if results_df is not None:
        return sql, results_df
    return sql, pd.DataFrame()


def main():
    setup_database()

    demo = gr.Interface(
        fn=gradio_text2sql,
        inputs=gr.Textbox(
            label="Ask a question about your data",
            placeholder="e.g., What are the top 5 products by rating?"
        ),
        outputs=[
            gr.Code(label="Generated SQL", language="sql"),
            gr.Dataframe(label="Query Results")
        ],
        title="Text-to-SQL Generator",
        description="Enter a natural language question and get SQL + results from the e-commerce database.",
    )

    demo.launch(server_name="0.0.0.0")


if __name__ == '__main__':
    main()
