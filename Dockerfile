FROM continuumio/miniconda3

WORKDIR /home/app

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app


CMD gunicorn api:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker 