FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY start_api.py .
COPY run_metadata.json .

RUN mkdir -p mlruns reports data

EXPOSE 8000

CMD ["python", "start_api.py"]
