FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# constraints.txt pins every package to the versions of the last known-good image (Aug 2025);
# newer langchain releases removed imports this code uses.
COPY requirements.txt constraints.txt ./

RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
