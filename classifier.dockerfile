# Dockerfile-gpu
FROM python:3.7
# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python-dev && \
     rm -rf /var/lib/apt/lists/*


RUN mkdir /app 
RUN mkdir /app/datasets
COPY pyproject.toml poetry.lock  /app

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

RUN pip install poetry
RUN poetry install --no-dev --no-root

COPY ./classifier /app/classifier
RUN poetry install --no-dev

ENV COMET_API_KEY ${COMET_API_KEY}

ENTRYPOINT ["poetry", "run", "python", "classifier/task.py"]
