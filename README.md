RAG console app
## Installation

Use the [docker](https://www.docker.com/) to install RAG console application.

Build RAG app image

```bash
docker build -t rag-llm-app .
```

Run builded image, replace /path/to/data with path to a folder where are stored documents you want to use in app

```bash
docker run -dit --network host -v /path/to/data:/app/data --name my-rag  rag-llm-app
```

Now you can run container bash and execute needed commands

```bash
docker exec -it my-rag bash
```

## Supported commands


Reset chromadb data

```bash
python populate_database.py --reset
```

Load data from your directory to chromadb

```bash
python populate_database.py
```

Search using imported data

```bash
python query_data "your_query"
```