FROM python:3.7.9

RUN mkdir /app 
COPY ./trainer /app/trainer
COPY pyproject.toml /app

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD} 

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "trainer/task.py"]
