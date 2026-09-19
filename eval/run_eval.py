"""
Runs the agent against eval/test_set.json and scores the answers.

Fill in:
  - A real scoring method: start with simple keyword/semantic overlap,
    then upgrade to RAGAS (faithfulness, answer relevancy) once
    agent/graph.py actually produces real answers.

Run standalone:
    python eval/run_eval.py
"""

import json
from pathlib import Path

TEST_SET_PATH = Path(__file__).parent / "test_set.json"


def load_test_set() -> list[dict]:
    return json.loads(TEST_SET_PATH.read_text())


def score_answer(question: str, expected: str, actual: str) -> float:
    """
    TODO: replace with a real metric. Placeholder just checks for
    non-empty output so you can confirm the eval loop runs end-to-end.
    """
    return 1.0 if actual.strip() else 0.0


def run_eval():
    test_set = load_test_set()
    results = []
    for case in test_set:
        # TODO: call your actual agent here instead of this placeholder
        actual = "TODO: call agent.graph.run_pipeline(...) and extract its answer"
        score = score_answer(case["question"], case["expected_answer"], actual)
        results.append({"question": case["question"], "score": score})

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"Average score: {avg:.2f}")
    for r in results:
        print(f"  [{r['score']:.1f}] {r['question']}")


if __name__ == "__main__":
    run_eval()
