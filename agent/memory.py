"""
Simple conversation memory. Start here; swap for LangGraph's built-in
state/checkpointing once you're comfortable with the basics.

Run standalone:
    python -m agent.memory
"""


class ConversationMemory:
    def __init__(self):
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def as_messages(self) -> list[dict]:
        """Return history in a shape your LLM client expects."""
        return self.history

    def clear(self):
        self.history = []


if __name__ == "__main__":
    mem = ConversationMemory()
    mem.add("user", "What happened to AAPL today?")
    mem.add("assistant", "placeholder response")
    print(mem.as_messages())
