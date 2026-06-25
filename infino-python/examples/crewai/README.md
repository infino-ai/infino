# CrewAI + Infino examples

Build CrewAI agents on [Infino](https://pypi.org/project/infino/) through the
[`crewai-infino`](https://pypi.org/project/crewai-infino/) integration — give a
crew **semantic (vector), full-text (BM25), hybrid, and SQL** retrieval as tools
over **one** store on object storage. No second vector database, no separate
metadata store, no client-side fusion: one Infino table answers all four ways,
and the agent picks the right one at runtime.

The examples use a **real public dataset** (HuggingFace Hub) or inline content,
with the same local `all-MiniLM-L6-v2` embeddings as the rest of the suite, so
building and searching the store runs **locally and key-free**. The crew itself is tool-calling, so it
needs an LLM key; each notebook says so up front and skips the crew without one.

## Examples

| Notebook | What it shows | LLM |
| -------- | ------------- | --- |
| [`01_support_triage_crew.ipynb`](01_support_triage_crew.ipynb) | A researcher → resolver crew over **one** Infino store: it answers an error code by exact keyword (BM25), a "how do I…" by meaning (vector), and prices/counts by SQL — then writes the customer reply. | required |
| [`02_content_research_crew.ipynb`](02_content_research_crew.ipynb) | A researcher → writer crew that drafts a short, **cited** explainer: the researcher grounds every point in a real document via hybrid search, the writer cites sources by title — no hallucinated sources. | required |
| [`03_knowledge_crew.ipynb`](03_knowledge_crew.ipynb) | The easy path: hand the crew an `InfinoKnowledgeStorage` and it **auto-grounds** every answer — no tools to wire, no retrieval calls. A dozen lines, one table. | required |

## Setup

```sh
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding model (~90 MB) and a dataset sample, so
the first cell can take a minute; later runs use the cache.

### Optional: LLM answers

The notebooks pick up a key automatically (via `_shared/llm.py`'s env handling),
reading a local `.azure.env` / `.env` if present:

- **Azure AI Foundry** (preferred): `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`, `DEFAULT_AZURE_MODEL`.
- **OpenAI**: `OPENAI_API_KEY` (optionally `OPENAI_MODEL`).

Keep credentials in an untracked env file — never commit keys.

## How this maps to CrewAI

Two seams. **Tools** are opt-in: `index.as_tools()` returns standard
`crewai.tools.BaseTool`s (semantic, keyword, hybrid, SQL) any `Agent` can call.
**Knowledge** is automatic: `InfinoKnowledgeStorage` plugs into CrewAI's
`BaseKnowledgeStorage`, so a crew grounds answers from one Infino table with no
tool wiring. The shared glue lives one level up in [`../_shared/`](../_shared/): `crew.py`
builds an optional CrewAI `LLM` from the environment; `loaders.py`, `embedding.py`,
and `llm.py` are reused unchanged from the base suite.
