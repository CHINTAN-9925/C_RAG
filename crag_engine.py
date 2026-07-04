"""
Corrective RAG (CRAG) engine.

This is a faithful port of the logic in `crag.ipynb`, refactored so the
document source is a set of user-uploaded PDFs instead of hard-coded files.

Pipeline (LangGraph):
    retrieve -> eval_each_doc -> [route]
        CORRECT             -> refine -> generate
        INCORRECT/AMBIGUOUS -> rewrite_query -> web_search -> refine -> generate

Knowledge refinement policy:
    CORRECT   => internal docs only
    INCORRECT => web docs only
    AMBIGUOUS => internal + web
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import List, TypedDict

from pydantic import BaseModel, Field, AliasChoices, ConfigDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LLM / embedding providers ---------------------------------------------
# Free stack (active): Groq for chat + a local HuggingFace model for embeddings.
# Neither costs anything to run (embeddings run locally; Groq has a free tier).
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Paid stack (OpenAI). Kept for reference. To switch back, uncomment the import
# below and the two usages marked "OpenAI (paid)" in build_index/build_graph,
# then comment out the Groq/HuggingFace equivalents.
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# ---------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

# Tavily is optional. If the package or API key is missing we skip web search.
# Prefer the maintained langchain-tavily package; fall back to the deprecated
# community tool if it is not installed.
try:
    from langchain_tavily import TavilySearch as _TavilyTool
except Exception:  # pragma: no cover
    try:
        from langchain_community.tools.tavily_search import (
            TavilySearchResults as _TavilyTool,
        )
    except Exception:
        _TavilyTool = None


# -----------------------------
# Thresholds (from the notebook)
# -----------------------------
UPPER_TH = 0.7
LOWER_TH = 0.3


# -----------------------------
# Graph state
# -----------------------------
class State(TypedDict):
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    web_docs: List[Document]

    answer: str


# -----------------------------
# Structured-output schemas
# -----------------------------
class DocEvalScore(BaseModel):
    # Accept common key variants the model may emit (e.g. "relevance_score").
    model_config = ConfigDict(populate_by_name=True)
    score: float = Field(
        validation_alias=AliasChoices("score", "relevance_score", "relevance")
    )
    reason: str = ""


class KeepOrDrop(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    keep: bool = Field(validation_alias=AliasChoices("keep", "relevant", "keep_sentence"))


class WebQuery(BaseModel):
    query: str


def _decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def build_index(pdf_files, chunk_size: int = 900, chunk_overlap: int = 150, k: int = 4):
    """
    Build a FAISS retriever from uploaded PDF files.

    `pdf_files` is a list of (filename, bytes) tuples.
    Returns (retriever, num_chunks).
    """
    docs: List[Document] = []
    for name, data in pdf_files:
        # PyPDFLoader needs a real path, so stage the bytes in a temp file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            loaded = PyPDFLoader(tmp_path).load()
            for d in loaded:
                d.metadata["source"] = name
            docs.extend(loaded)
        finally:
            os.unlink(tmp_path)

    if not docs:
        raise ValueError("No text could be extracted from the uploaded PDF(s).")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Free: local sentence-transformers model (downloads once, ~90MB, no API key).
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # OpenAI (paid) alternative:
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever, len(chunks)


def build_graph(retriever):
    """Compile and return the CRAG LangGraph app bound to a given retriever."""
    # Free: Groq (needs a free GROQ_API_KEY). Supports the structured output
    # the grading / filter / rewrite nodes rely on.
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # OpenAI (paid) alternative:
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    web_enabled = _TavilyTool is not None and bool(os.getenv("TAVILY_API_KEY"))
    tavily = _TavilyTool(max_results=5) if web_enabled else None

    # --- Retrieve ---
    def retrieve_node(state: State) -> State:
        return {"docs": retriever.invoke(state["question"])}

    # --- Score-based per-doc evaluator ---
    doc_eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict retrieval evaluator for RAG.\n"
                "You will be given ONE retrieved chunk and a question.\n"
                "Return a relevance score in [0.0, 1.0].\n"
                "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
                "- 0.0: chunk is irrelevant\n"
                "Be conservative with high scores.\n"
                'Output JSON only with EXACTLY these keys: '
                '"score" (a number in [0.0, 1.0]) and "reason" (a short string). '
                'Do not rename the keys.',
            ),
            ("human", "Question: {question}\n\nChunk:\n{chunk}"),
        ]
    )
    # json_mode (not tool-calling): Llama on Groq sometimes emits strings like
    # "0.5" / "false" for typed fields, which Groq's tool validator rejects.
    # json_mode returns raw JSON that Pydantic then coerces to the right types.
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(
        DocEvalScore, method="json_mode"
    )

    def eval_each_doc_node(state: State) -> State:
        q = state["question"]
        scores: List[float] = []
        good: List[Document] = []

        for d in state["docs"]:
            out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
            scores.append(out.score)
            if out.score > LOWER_TH:
                good.append(d)

        if any(s > UPPER_TH for s in scores):
            return {
                "good_docs": good,
                "verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
            }
        if len(scores) > 0 and all(s < LOWER_TH for s in scores):
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {LOWER_TH}.",
            }
        return {
            "good_docs": good,
            "verdict": "AMBIGUOUS",
            "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
        }

    # --- Sentence-level filter (LLM judge) ---
    filter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict relevance filter.\n"
                "Keep the sentence only if it directly helps answer the question.\n"
                "Use ONLY the sentence.\n"
                'Output JSON only with EXACTLY one key: "keep" (a boolean). '
                'Do not rename the key.',
            ),
            ("human", "Question: {question}\n\nSentence:\n{sentence}"),
        ]
    )
    filter_chain = filter_prompt | llm.with_structured_output(
        KeepOrDrop, method="json_mode"
    )

    def refine(state: State) -> State:
        q = state["question"]
        verdict = state.get("verdict")

        if verdict == "CORRECT":
            docs_to_use = state["good_docs"]
        elif verdict == "INCORRECT":
            docs_to_use = state.get("web_docs", [])
        else:  # AMBIGUOUS
            docs_to_use = state["good_docs"] + state.get("web_docs", [])

        context = "\n\n".join(d.page_content for d in docs_to_use).strip()
        strips = _decompose_to_sentences(context)

        kept: List[str] = []
        for s in strips:
            if filter_chain.invoke({"question": q, "sentence": s}).keep:
                kept.append(s)

        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": "\n".join(kept).strip(),
        }

    # --- Query rewrite for web search ---
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user question into a web search query composed of keywords.\n"
                "Rules:\n"
                "- Keep it short (6-14 words).\n"
                "- If the question implies recency, add a constraint like (last 30 days).\n"
                "- Do NOT answer the question.\n"
                "- Return JSON with a single key: query",
            ),
            ("human", "Question: {question}"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm.with_structured_output(
        WebQuery, method="json_mode"
    )

    def rewrite_query_node(state: State) -> State:
        out = rewrite_chain.invoke({"question": state["question"]})
        return {"web_query": out.query}

    def web_search_node(state: State) -> State:
        if tavily is None:
            return {"web_docs": []}
        q = state.get("web_query") or state["question"]
        try:
            raw = tavily.invoke({"query": q})
        except Exception:
            return {"web_docs": []}

        # Normalize the various shapes Tavily can return:
        #   - langchain_tavily.TavilySearch -> {"results": [ {...}, ... ], ...}
        #   - deprecated TavilySearchResults -> [ {...}, ... ]
        #   - error / edge cases -> a plain string
        if isinstance(raw, dict):
            results = raw.get("results", [])
        elif isinstance(raw, str):
            return {"web_docs": [Document(page_content=raw, metadata={})]}
        else:
            results = raw or []

        web_docs: List[Document] = []
        for r in results:
            if isinstance(r, str):
                web_docs.append(Document(page_content=r, metadata={}))
                continue
            if not isinstance(r, dict):
                continue
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
        return {"web_docs": web_docs}

    # --- Generate ---
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful tutor. Answer ONLY using the provided context.\n"
                "If the context is empty or insufficient, say: 'I don't know.'",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    def generate(state: State) -> State:
        out = (answer_prompt | llm).invoke(
            {"question": state["question"], "context": state["refined_context"]}
        )
        return {"answer": out.content}

    # --- Routing ---
    def route_after_eval(state: State) -> str:
        return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"

    # --- Build graph ---
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {"refine": "refine", "rewrite_query": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()


def empty_state(question: str) -> State:
    return {
        "question": question,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }


def answer_question(graph, question: str) -> State:
    """Run the compiled CRAG graph for a single question."""
    return graph.invoke(empty_state(question))
