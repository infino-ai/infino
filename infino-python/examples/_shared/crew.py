"""CrewAI glue for the examples: an optional LLM from the same env convention.

`crew_llm()` returns a configured CrewAI `LLM` (Azure AI Foundry or OpenAI, same
env as `_shared/llm.py`) or `None` when no key is set — callers degrade to
running the Infino retrieval tools directly, without an agent.
"""

import os

from _shared.llm import _load_env_files  # reuse the suite's .env loader


def crew_llm(temperature: float = 0.0):
    """A CrewAI chat model from the environment, or `None` if no key is set.

    Prefers Azure AI Foundry (`AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`,
    `DEFAULT_AZURE_MODEL`) over OpenAI (`OPENAI_API_KEY`, optional `OPENAI_MODEL`),
    matching `_shared/llm.py`. The `openai/` prefix routes both through CrewAI's
    OpenAI-compatible path (the Azure endpoint is an OpenAI-compatible URL).
    """
    _load_env_files()
    from crewai import LLM

    azure_endpoint = os.environ.get("AZURE_AI_ENDPOINT")
    azure_key = os.environ.get("AZURE_AI_API_KEY")
    if azure_endpoint and azure_key:
        model = os.environ.get("DEFAULT_AZURE_MODEL", "gpt-4o-mini")
        return LLM(
            model=f"openai/{model}", base_url=azure_endpoint, api_key=azure_key,
            temperature=temperature,
        )

    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return LLM(model=f"openai/{model}", temperature=temperature)

    return None
