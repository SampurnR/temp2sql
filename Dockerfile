FROM python:3.11-slim

WORKDIR /app

COPY code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY code/ ./code/

WORKDIR /app/code

EXPOSE 7860

CMD ["python", "main.py"]

# docker build . -t text2sql:latest
# docker run -p 7860:7860 -e GOOGLE_API_KEY=your-key text2sql