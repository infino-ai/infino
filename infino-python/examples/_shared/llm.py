"""Optional LLM answer generation.

Uses Azure AI Foundry (`AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`,
`DEFAULT_AZURE_MODEL`) or OpenAI (`OPENAI_API_KEY`, optional `OPENAI_MODEL`);
`complete()` returns `None` when neither is set. Credentials come from the
environment or a local `.env` / `.azure.env`.
"""

import os
from pathlib import Path

_ENV_FILES = (".azure.env", ".env")
_loaded = False


def _load_env_files() -> None:
    """Load the nearest .azure.env / .env into os.environ once; existing vars win."""
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


def complete(prompt: str) -> str | None:
    """Answer `prompt` with the configured LLM, or return None if none is set."""
    client, model = _client_and_model()
    if client is None:
        return None
    reply = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return reply.choices[0].message.content
