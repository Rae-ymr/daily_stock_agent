"""
Corrective RAG: retrieve, grade relevance, retry with a rewritten query
if the retrieval was weak. This is the piece to build and test FIRST,
in isolation, before wiring it into the full agent graph.

Requires GROQ_API_KEY (or OPENAI_API_KEY, see agent/llm.py) in .env for the
grade/rewrite LLM calls. Embeddings run locally via sentence-transformers —
no API key needed.

Run standalone:
    python -m agent.retrieval "What happened to AAPL this week?"
"""

import sys

from data.ingest_news import fetch_news
from agent.llm import get_chat_llm

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

MAX_RETRIES = 2
_llm = get_chat_llm()
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(documents: list[dict]):
    """
    Embeds `documents` (from ingest_news.py output — {title, source, url,
    published_at} dicts) and stores them in an in-memory Chroma store.
    Headlines are short, so each article is stored as one Document —
    no chunking needed. Returns the store for retrieve() to search.
    """
    docs = [
        Document(
            page_content=doc["title"],
            metadata={
                "source": doc.get("source", ""),
                "url": doc.get("url", ""),
                "published_at": doc.get("published_at", ""),
            },
        )
        for doc in documents
    ]
    return Chroma.from_documents(docs, embedding=_embeddings)


def retrieve(query: str, store, k: int = 5) -> list[dict]:
    """Runs similarity search against `store`, returns top-k chunks as dicts."""
    results = store.similarity_search(query, k=k)
    return [{"content": doc.page_content, **doc.metadata} for doc in results]


def grade_relevance(query: str, chunks: list[dict]) -> bool:
    """
    Asks an LLM "do these chunks actually answer this query?" Keeps the
    prompt simple — this only needs to catch clearly bad retrievals, not
    make a fine-grained judgment call.
    """
    if not chunks:
        return False
    headlines = "\n".join(f"- {c['content']}" for c in chunks)
    prompt = (
        f"Question: {query}\n\nRetrieved headlines:\n{headlines}\n\n"
        "Do these headlines contain information relevant to answering the "
        "question? Reply with only 'yes' or 'no'."
    )
    response = _llm.invoke(prompt)
    return response.content.strip().lower().startswith("y")


def rewrite_query(query: str) -> str:
    """Asks an LLM to rephrase the query for a better search hit."""
    prompt = (
        f"This search query returned weak results: \"{query}\"\n"
        "Rewrite it as a clearer, more specific search query for finding "
        "recent news about the same topic. Reply with only the new query."
    )
    response = _llm.invoke(prompt)
    return response.content.strip()


def retrieve_with_retry(query: str, store) -> list[dict]:
    """
    The main entry point other modules should call. Retrieves, grades,
    and retries with a rewritten query up to MAX_RETRIES times before
    giving up and returning whatever it last found.
    """
    current_query = query
    for attempt in range(MAX_RETRIES + 1):
        chunks = retrieve(current_query, store)
        if grade_relevance(query, chunks):
            return chunks
        print(f"[retry {attempt + 1}] weak retrieval, rewriting query")
        current_query = rewrite_query(current_query)
    return chunks  # last attempt, even if weak


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    query = f"What happened to {ticker} this week?"

    documents = fetch_news(ticker)
    store = build_vector_store(documents)
    results = retrieve_with_retry(query, store)

    print(f"\nQuery: {query}")
    for r in results:
        print(f"- {r['content']} ({r.get('source', 'unknown')})")
