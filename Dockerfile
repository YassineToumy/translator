FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY translator.py .

RUN mkdir -p /app/logs

CMD ["python", "-u", "translator.py"]
