"""
Runs the agent against eval/test_set.json and scores the answers.

Scoring: cosine similarity between the agent's draft and expected_answer,
using the same embedding model agent/retrieval.py already uses. This is
the "semantic overlap" tier from the plan — good enough to catch "the
agent talked about the wrong thing entirely" without needing an LLM
judge. Upgrade to RAGAS (faithfulness / answer_relevancy over
retrieved_context, already in requirements.txt) once test_set.json has
enough real cases to make that worthwhile.

Run standalone:
    python -m eval.run_eval
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from agent.graph import (
    decision_node,
    draft_summary_node,
    ingest_node,
    intel_node,
    retrieve_and_grade_node,
    risk_node,
    technical_node,
)
from agent.retrieval import build_vector_store

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
PLACEHOLDER_MARKER = "REPLACE WITH"

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_test_set() -> list[dict]:
    return json.loads(TEST_SET_PATH.read_text())


def run_agent(ticker: str) -> dict:
    """
    Runs the pipeline through draft_summary_node, one stage at a time.
    Deliberately skips human_checkpoint_node — it blocks on terminal
    input for approval, which isn't something an automated eval loop
    should hit. Eval only needs the drafted analysis, not the approval.
    """
    state = {"ticker": ticker}
    state.update(ingest_node(state))
    state["store"] = build_vector_store(state["news"])
    state.update(retrieve_and_grade_node(state))
    state.update(technical_node(state))
    state.update(intel_node(state))
    state.update(risk_node(state))
    state.update(decision_node(state))
    state.update(draft_summary_node(state))
    return state


def score_answer(expected: str, actual: str) -> Optional[float]:
    """
    Cosine similarity between expected and actual answer embeddings.
    Returns None when expected_answer is still the placeholder text,
    so run_eval can report "not scored yet" instead of a misleading 0.0.
    """
    if PLACEHOLDER_MARKER in expected:
        return None
    if not actual.strip():
        return 0.0
    expected_vec, actual_vec = _embeddings.embed_documents([expected, actual])
    a, b = np.array(expected_vec), np.array(actual_vec)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_eval():
    test_set = load_test_set()
    results = []
    for case in test_set:
        ticker = case.get("ticker")
        if not ticker:
            print(f"Skipping (no \"ticker\" field): {case['question']}")
            continue
        state = run_agent(ticker)
        actual = state.get("draft", "")
        score = score_answer(case["expected_answer"], actual)
        results.append(
            {"question": case["question"], "score": score, "actual": actual}
        )

    scored = [r for r in results if r["score"] is not None]
    avg = sum(r["score"] for r in scored) / len(scored) if scored else 0.0

    print(f"\nAverage similarity score: {avg:.2f} ({len(scored)}/{len(results)} cases scored)\n")
    for r in results:
        label = f"{r['score']:.2f}" if r["score"] is not None else "not scored (placeholder expected_answer)"
        print(f"[{label}] {r['question']}")
        print(f"    -> {r['actual'][:200]}\n")


if __name__ == "__main__":
    run_eval()
