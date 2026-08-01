# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors

"""Print the example notebooks that ``make python-examples-test`` should run.

The ``langchain/`` and ``crewai/`` suites run through the published integration
packages (``langchain-infino`` / ``crewai-infino``), which the test installs
with ``--no-deps`` and links against the from-source infino build. A breaking
infino API change can land before those packages ship a compatible release —
they cannot depend on the new infino until it is published — so during that
window their notebooks would hard-fail through no fault of this repo.

To avoid that, each integration is gated on a *compat probe* that exercises the
integration's own table-creation path against an in-memory infino. If the probe
raises (e.g. an ``IndexSpec`` signature change), the integration is not yet
compatible: its notebooks are skipped with a note to stderr and omitted from the
printed list. The gate self-heals — once the integration releases a compatible
version the probe passes and its notebooks run again. Direct-infino examples
(``rag/``, ``code_search/``, ...) are never gated.

Stdout: the notebooks to execute, one path per line.
"""

from __future__ import annotations

import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)

# A tiny vector width — the probe only needs to reach the index-spec build.
_DIM = 8


def _probe_langchain() -> None:
    import infino
    from langchain_core.embeddings import Embeddings
    from langchain_infino import InfinoVectorStore

    class _FixedEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.1] * _DIM for _ in texts]

        def embed_query(self, text):
            return [0.1] * _DIM

    conn = infino.connect("memory://")
    InfinoVectorStore.from_texts(
        ["compat probe"],
        _FixedEmbeddings(),
        connection=conn,
        table_name="_compat_probe",
        dim=_DIM,
    )


def _probe_crewai() -> None:
    import infino
    from crewai_infino import InfinoIndex

    conn = infino.connect("memory://")
    InfinoIndex.create(
        conn,
        "_compat_probe",
        embed_documents=lambda texts: [[0.1] * _DIM for _ in texts],
        embed_query=lambda text: [0.1] * _DIM,
        dim=_DIM,
    )


# example subdirectory -> the probe that decides whether to run it
_GATED = {"langchain": _probe_langchain, "crewai": _probe_crewai}


def _compatible(name: str) -> bool:
    try:
        _GATED[name]()
    except Exception as exc:  # noqa: BLE001 — any failure ⇒ treat as incompatible
        print(
            f"note: skipping {name} examples — {name}-infino not compatible with "
            f"this infino build ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return False
    return True


def main() -> None:
    skip = {name for name in _GATED if not _compatible(name)}
    for nb in sorted(glob.glob(os.path.join(_EXAMPLES, "*", "[0-9]*.ipynb"))):
        if os.path.basename(os.path.dirname(nb)) in skip:
            continue
        print(nb)


if __name__ == "__main__":
    main()
