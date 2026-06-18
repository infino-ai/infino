"""Optional LLM answer generation for the RAG examples.

Resolves credentials in this order, so the examples run anywhere:

1. **Azure AI Foundry** (preferred) — `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`,
   `DEFAULT_AZURE_MODEL`. The endpoint is the OpenAI-compatible `/openai/v1`
   path, so the standard `openai` client talks to it directly.
2. **OpenAI** — `OPENAI_API_KEY` (optionally `OPENAI_MODEL`).
3. **Neither** — `complete()` returns `None`, and callers fall back to showing
   the retrieved sources, so every notebook still runs key-free.

Credentials are read from the environment, or from a `.env`/`.azure.env` file
found by walking up from this directory (never hard-coded, never committed —
those files are git-ignored).
"""

import os
from pathlib import Path

_ENV_FILES = (".azure.env", ".env")
_loaded = False


def _load_env_files() -> None:
    """Populate os.environ from the nearest .azure.env / .env, once.

    Walks up from this file to the filesystem root. Existing environment
    variables win, so an explicitly exported value is never overridden.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    for parent in Path(__file__).resolve().parents:
        for name in _ENV_FILES:
            path = parent / name
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _client_and_model():
    """Return (openai.OpenAI, model_name) or (None, None) if no credentials."""
    _load_env_files()
    from openai import OpenAI

    azure_endpoint = os.environ.get("AZURE_AI_ENDPOINT")
    azure_key = os.environ.get("AZURE_AI_API_KEY")
    if azure_endpoint and azure_key:
        model = os.environ.get("DEFAULT_AZURE_MODEL", "gpt-4o-mini")
        return OpenAI(base_url=azure_endpoint, api_key=azure_key), model

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    return None, None


def have_llm() -> bool:
    """True if an LLM backend is configured."""
    client, _ = _client_and_model()
    return client is not None


def complete(prompt: str, system: str | None = None) -> str | None:
    """Generate a completion, or return None if no LLM is configured."""
    client, model = _client_and_model()
    if client is None:
        return None
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    reply = client.chat.completions.create(model=model, messages=messages)
    return reply.choices[0].message.content
