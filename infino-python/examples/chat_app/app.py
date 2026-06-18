"""Chat with your documents — a Streamlit app backed by Infino.

Run it:

    streamlit run app.py

Ingest a seed Wikipedia corpus or upload your own .txt / .pdf files, then ask
questions. Retrieval is hybrid (BM25 + vector) fused in-engine; answers cite
their sources. The document index is a durable Infino table on local disk, so
it survives restarts — no server, no separate vector database.

Set OPENAI_API_KEY to get generated answers; without it, the app shows the
retrieved source passages (still useful, and key-free).
"""

import streamlit as st

# Importing rag also puts examples/ on sys.path, so `_shared` is importable below.
import rag

# pypdf is only needed if the user uploads PDFs.
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


@st.cache_resource
def get_db():
    """One durable connection per session (cached across Streamlit reruns)."""
    db = rag.open_db()
    rag.get_table(db)  # ensure the table exists
    return db


def read_upload(file) -> str:
    if file.name.lower().endswith(".pdf"):
        if PdfReader is None:
            st.error("Install pypdf to ingest PDFs: pip install pypdf")
            return ""
        return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
    return file.read().decode("utf-8", errors="ignore")


st.set_page_config(page_title="Chat with your docs · Infino", page_icon="📄")
st.title("📄 Chat with your documents")
st.caption("Hybrid search (BM25 + vector) over one Infino table on local disk.")

db = get_db()

with st.sidebar:
    st.header("Document index")
    st.metric("Indexed chunks", rag.count(db))

    st.subheader("Seed corpus")
    n_seed = st.slider("Wikipedia articles", 10, 300, 50, step=10)
    if st.button("Ingest Wikipedia sample"):
        from _shared.datasets import load_wikipedia

        with st.spinner("Downloading and indexing…"):
            added = rag.ingest(rag.get_table(db), load_wikipedia(n=n_seed))
        st.success(f"Indexed {added} chunks")
        st.rerun()

    st.subheader("Your files")
    uploads = st.file_uploader(
        "Upload .txt or .pdf", type=["txt", "pdf"], accept_multiple_files=True
    )
    if uploads and st.button("Ingest uploads"):
        docs = [
            {"title": f.name, "text": read_upload(f), "source": f.name}
            for f in uploads
        ]
        docs = [d for d in docs if d["text"].strip()]
        with st.spinner("Indexing…"):
            added = rag.ingest(rag.get_table(db), docs)
        st.success(f"Indexed {added} chunks from {len(docs)} files")
        st.rerun()

if rag.count(db) == 0:
    st.info("No documents indexed yet — use the sidebar to ingest a corpus or upload files.")
    st.stop()

question = st.chat_input("Ask a question about your documents")
if question:
    with st.chat_message("user"):
        st.write(question)

    hits = rag.retrieve(db, question, k=4)
    with st.chat_message("assistant"):
        generated = rag.answer(question, hits)
        if generated:
            st.write(generated)
        else:
            st.write("_No `OPENAI_API_KEY` set — showing the retrieved sources:_")
        with st.expander("Sources", expanded=not generated):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[{i}] {h['title']}** · `{h['source']}` · score {h['score']:.3f}")
                st.write(h["text"])
