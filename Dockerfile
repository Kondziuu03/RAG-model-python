FROM python:3.12.3-slim

WORKDIR /app

RUN export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir ./data
RUN touch ./data/chat_history.sqlite3

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_rag.py", "--server.address", "0.0.0.0"]
