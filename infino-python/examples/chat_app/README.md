# Chat with your documents

A small [Streamlit](https://streamlit.io) app that lets you **chat with your own documents** using [Infino](https://pypi.org/project/infino/) for retrieval. It ties together the techniques from the notebooks in this directory:

- **Hybrid retrieval** (BM25 + vector, fused in one in-engine SQL query)
- **Cited sources** for every answer
- A **durable local index** — your documents survive app restarts
- **One engine, no server** — no separate vector database to run or sync

## Run it

```sh
pip install -r ../requirements.txt
streamlit run app.py
```

Then, in the sidebar:

1. **Ingest a seed corpus** — pull a sample of Wikipedia articles, or
2. **Upload your own** `.txt` / `.pdf` files.

Ask a question in the chat box. Answers cite the source passages they used.

> Set `OPENAI_API_KEY` to get generated answers. Without it, the app still
> works — it shows the retrieved source passages, key-free.

## How it works

| File | Role |
| --- | --- |
| `rag.py` | All the Infino logic — connect, ingest (chunk + embed + append), hybrid retrieve, prompt assembly. Importable and testable on its own. |
| `app.py` | The Streamlit UI — sidebar ingestion, chat input, source display. |

The document index is an Infino table stored under `chat_data/` (git-ignored).
Embeddings are computed locally with `sentence-transformers` — no API key needed
to build or query the index.
