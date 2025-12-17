FROM python:3.11-slim

WORKDIR /app

# Install OS deps for OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgl1-mesa-glx \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
