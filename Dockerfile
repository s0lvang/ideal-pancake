# Dockerfile-gpu
FROM echoesai/tensorflow-py3.7:latest-gpu
RUN ls -la
RUN rm /etc/apt/sources.list.d/cuda.list
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

COPY ./trainer /app/trainer
RUN poetry install --no-dev


ENTRYPOINT ["poetry", "run", "python", "trainer/task.py"]
