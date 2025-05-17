# RAG Model Python

A Streamlit application for document search and question answering using RAG (Retrieval-Augmented Generation) with multiple LLM providers.

## Features

- Multiple LLM providers support (OpenAI, Ollama, PG Bielik)
- PDF document processing and chunking
- Vector similarity search using ChromaDB
- Session management for multiple document sets
- Interactive chat interface
- Model comparison in "Arena" mode

# Setup
## RAG container
Build RAG app image

```bash
docker build -t streamlit-app .
```
Run builded image

```bash
docker run -p 8501:8501 streamlit-app
```
## Ollama container

Pull Ollama image

```bash
docker pull ollama/ollama
```

Run the Ollama container

```bash
docker run -d --name my-ollama -p 11434:11434 ollama/ollama
```

## Use docker compose instead

Build image and start

```bash
docker compose up --build
```

Access the ollama container

```bash
docker compose exec ollama bash
```
Pull needed LLM models

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text
```
## Usage

1. Start by creating a new session in the "Sesje" tab
2. Upload PDF documents in the "Dokumenty" tab
3. Configure provider settings in the "Ustawienia" tab
4. Start chatting with your documents in the "Czat" tab
5. Compare different models in the "Arena" tab
