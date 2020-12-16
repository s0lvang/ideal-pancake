# Dockerfile-gpu
FROM echoesai/tensorflow-py3.7:latest-gpu

RUN rm /etc/apt/sources.list.d/cuda.list
# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python-dev && \
     rm -rf /var/lib/apt/lists/*

RUN mkdir /app 
COPY pyproject.toml poetry.lock  /app

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN pip install poetry
RUN poetry install --no-dev --no-root

COPY ./trainer /app/trainer
RUN poetry install --no-dev


ENTRYPOINT ["poetry", "run", "python", "trainer/task.py"]
