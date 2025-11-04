FROM python:3.11-slim

WORKDIR /app

# System deps (build tools in case wheels are missing)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

ENV PORT=5000
EXPOSE 5000

# Use Gunicorn to serve Flask app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]


