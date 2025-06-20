FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y gcc g++ libpq-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
