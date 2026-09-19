"""
Redis-backed store for in-progress human-checkpoint sessions.

Holds only the slice of AgentState needed to resume after a human
approves or rejects a draft over HTTP — not the full LangGraph state.
The Chroma vector store in AgentState isn't JSON-serializable and isn't
needed past draft_summary_node anyway, so it's dropped before storing.

Run standalone (needs Redis reachable via REDIS_URL, default localhost):
    python -m app.session_store
"""

import json
import os
from typing import Optional, TypedDict
from uuid import uuid4

import redis

SESSION_TTL_SECONDS = 1800  # abandoned sessions (draft shown, never approved) expire after 30 min

_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
)


class SessionState(TypedDict):
    ticker: str
    price_summary: str
    analysis: str
    draft: str
    revision_count: int


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def create_session(state: SessionState) -> str:
    """Stores `state` under a fresh session_id and returns it."""
    session_id = str(uuid4())
    _client.setex(_key(session_id), SESSION_TTL_SECONDS, json.dumps(state))
    return session_id


def load_session(session_id: str) -> Optional[SessionState]:
    """Returns the stored state, or None if session_id is unknown/expired."""
    raw = _client.get(_key(session_id))
    return json.loads(raw) if raw else None


def update_session(session_id: str, state: SessionState) -> None:
    """Overwrites the stored state and refreshes its TTL."""
    _client.setex(_key(session_id), SESSION_TTL_SECONDS, json.dumps(state))


def delete_session(session_id: str) -> None:
    _client.delete(_key(session_id))


if __name__ == "__main__":
    sid = create_session(
        {"ticker": "AAPL", "price_summary": "test", "analysis": "test", "draft": "test draft", "revision_count": 0}
    )
    print(f"created session {sid}: {load_session(sid)}")
    delete_session(sid)
    print(f"after delete: {load_session(sid)}")
