# RAG app

# Installation

Use the [docker](https://www.docker.com/) to install RAG.

## RAG container

Build RAG app image

```bash
docker build -t streamlit-app .
```

Run builded image

```bash
docker run -p 8501:8501 --env-file .env streamlit-app
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