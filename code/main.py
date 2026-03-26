import os
import gradio as gr
from dotenv import load_dotenv
from google import genai

from helper import setup_database, text2sql

load_dotenv()

genai_client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])


def gradio_text2sql(user_query):
    """Wrapper for Gradio: returns both the SQL and the results table."""
    sql, results_df = text2sql(genai_client, user_query)
    if results_df is not None:
        return sql, results_df
    return sql, "No results or query error."


def main():
    # Set up the database if it doesn't exist
    if not os.path.exists('ecommerce.db'):
        setup_database(db_name='ecommerce.db', data_dir='../data')

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

    demo.launch()


if __name__ == '__main__':
    main()
