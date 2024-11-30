FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install poetry
RUN poetry install

CMD ["poetry", "run", "python", "src/app/__init__.py"]
