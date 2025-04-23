# 🧠 Local RAG with Ollama, LangChain & Chroma

A lightweight Retrieval-Augmented Generation (RAG) pipeline using:

- 🧠 [Ollama](https://ollama.com) — run LLMs like llama3.2 locally
- 🌐 LangChain — document and web URL loaders
- 🧊 Chroma — local vector database to store, index and retrieve embeddings for quick search purposes

---

## ⚙️ Installation

### 1. Install Ollama (if not already)

Follow the official instructions: https://ollama.com/download

Once installed, run a model locally:

```bash
ollama run llama3.2
```


### 2. Set up Python Environment

We recommend using a virtual environment:

```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Project structure
```
rag-ollama/
│
├── rag_runner.py         # The main RAG script
├── README.md             # This file
├── requirements.txt      # The required Python dependencies
└── chroma_db/            # Vector DB files (auto-created)
└── rag_env/              # Python virtual environment
```

### 4. Running the script

Edit rag_runner.py (example provided below), then run:

```bash
python rag_runner.py
```

This script will:
- Load content from a webpage
- Chunk and embed it
- Store in Chroma vector DB
- Run a query
- Generate an answer via Ollama
