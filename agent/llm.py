"""
Picks the chat LLM provider based on LLM_PROVIDER in .env.
Defaults to Groq (free tier, no cost for this project's usage).
Set LLM_PROVIDER=openai in .env to use OpenAI instead.
"""

import os


def get_chat_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    from langchain_groq import ChatGroq

    return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
