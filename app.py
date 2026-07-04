"""
Streamlit frontend for Corrective RAG (CRAG).

Upload one or more text PDFs, then ask questions. The CRAG graph grades the
retrieved chunks, corrects with web search when they are weak, refines the
context sentence-by-sentence, and answers from that refined context.

Run:  streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from crag_engine import build_index, build_graph, answer_question

load_dotenv()

st.set_page_config(page_title="Corrective RAG", layout="centered")

# --- Session state ---
st.session_state.setdefault("graph", None)
st.session_state.setdefault("num_chunks", 0)
st.session_state.setdefault("indexed_files", [])
st.session_state.setdefault("history", [])

VERDICT_LABEL = {
    "CORRECT": "Answered from your documents.",
    "AMBIGUOUS": "Documents were weak - combined with web search.",
    "INCORRECT": "Documents were irrelevant - answered from web search.",
}


def render_trace(reason, web_query, kept, strips, refined_context):
    with st.expander("Trace"):
        st.markdown(f"**Reason:** {reason}")
        if web_query:
            st.markdown(f"**Web query:** {web_query}")
        st.markdown(f"**Kept sentences:** {kept} of {strips}")
        if refined_context:
            st.text(refined_context)


def build_index_from(uploaded):
    with st.spinner("Reading PDFs and building index..."):
        try:
            pdf_files = [(f.name, f.getvalue()) for f in uploaded]
            retriever, n = build_index(pdf_files)
            st.session_state.graph = build_graph(retriever)
            st.session_state.num_chunks = n
            st.session_state.indexed_files = [f.name for f in uploaded]
            st.session_state.history = []
            st.rerun()
        except Exception as e:
            st.session_state.graph = None
            st.error(f"Failed: {e}")


# --- Sidebar: keys + status ---
with st.sidebar:
    st.header("Setup")

    if os.getenv("GROQ_API_KEY"):
        st.success("Groq key loaded")
    else:
        st.error("GROQ_API_KEY missing - add it to your .env file.")

    if os.getenv("TAVILY_API_KEY"):
        st.caption("Web-search correction: on")
    else:
        st.caption("Web-search correction: off (set TAVILY_API_KEY to enable)")

    st.caption("Embeddings run locally (HuggingFace) - no key needed.")

    if st.session_state.indexed_files:
        st.divider()
        st.caption(f"**Indexed ({st.session_state.num_chunks} chunks):**")
        for name in st.session_state.indexed_files:
            st.caption(f"- {name}")
        if st.button("Upload different PDFs", use_container_width=True):
            st.session_state.graph = None
            st.session_state.indexed_files = []
            st.session_state.history = []
            st.rerun()


# --- Main ---
st.title("Corrective RAG")
st.caption("Grades retrieval, self-corrects with web search, refines context, then answers.")

# Home page: upload + build lives right here in the main area.
if st.session_state.graph is None:
    st.subheader("Upload your PDFs")
    uploaded = st.file_uploader(
        "Drag and drop one or more text PDFs, then build the index.",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Build index", type="primary", disabled=not uploaded):
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY missing - add it to your .env file.")
        else:
            build_index_from(uploaded)

# Chat page: shown once an index exists.
else:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(f"{turn['verdict']} - {VERDICT_LABEL.get(turn['verdict'], '')}")
            render_trace(
                turn["reason"], turn.get("web_query"),
                turn["kept"], turn["strips"], turn.get("refined_context"),
            )

    question = st.chat_input("Ask a question about your PDFs...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving, grading, correcting, refining..."):
                try:
                    res = answer_question(st.session_state.graph, question)
                    st.write(res["answer"])
                    st.caption(
                        f"{res['verdict']} - {VERDICT_LABEL.get(res['verdict'], '')}"
                    )
                    render_trace(
                        res["reason"], res.get("web_query"),
                        len(res["kept_strips"]), len(res["strips"]),
                        res.get("refined_context"),
                    )
                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": res["answer"],
                            "verdict": res["verdict"],
                            "reason": res["reason"],
                            "web_query": res.get("web_query", ""),
                            "kept": len(res["kept_strips"]),
                            "strips": len(res["strips"]),
                            "refined_context": res.get("refined_context", ""),
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
