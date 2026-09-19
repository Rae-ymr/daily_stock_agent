"""
Pulls recent news headlines for a ticker, via Tavily's search API.

Fill in:
  - Rate limiting / caching if you're calling this often (Tavily's free
    tier is 1,000 requests/month — fine for daily dev use, but cache if
    you're iterating quickly during testing)

Run standalone:
    python -m data.ingest_news AAPL
"""

import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def fetch_news(ticker: str, limit: int = 10) -> list[dict]:
    """
    Returns a list of {title, source, url, published_at} for the given ticker.
    This is the shape agent/retrieval.py expects to embed.
    """
    if not TAVILY_API_KEY:
        print("TAVILY_API_KEY not set — returning placeholder data")
        return [
            {
                "title": f"Placeholder headline about {ticker}",
                "source": "placeholder",
                "url": "",
                "published_at": "",
            }
        ]

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=f"{ticker} stock",
        topic="news",
        days=14,
        max_results=limit,
    )

    articles = []
    for result in response.get("results", []):
        url = result.get("url", "")
        articles.append(
            {
                "title": result.get("title", ""),
                "source": urlparse(url).netloc.replace("www.", "") if url else "",
                "url": url,
                "published_at": result.get("published_date", ""),
            }
        )
    return articles


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    articles = fetch_news(ticker)
    for a in articles:
        print(f"- {a['title']} ({a['source']})")
