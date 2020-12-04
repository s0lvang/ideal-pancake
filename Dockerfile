# Dockerfile-gpu
FROM tiangolo/python-machine-learning:cuda9.1-python3.7

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python-dev && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*


RUN mkdir /app 
COPY pyproject.toml poetry.lock  /app

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN pip3 install poetry
RUN poetry env use 3.7.9
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-root

COPY ./trainer /app/trainer
RUN poetry install --no-dev

ENTRYPOINT ["python", "trainer/task.py"]
