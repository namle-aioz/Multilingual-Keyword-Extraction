FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirement.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

RUN echo "1.0.0" > keyword_list_version.txt

COPY multiple_extraction.py .
COPY data.csv .

CMD ["python", "multiple_extraction.py"]
