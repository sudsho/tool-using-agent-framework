"""Web search tool.

Three backends:

- ``tavily``  - simple POST to https://api.tavily.com/search
- ``serpapi`` - the SerpAPI Google endpoint
- ``mock``    - a small canned corpus, no network or keys (for offline runs
  and tests)

Pick a provider via the constructor; the two live backends read their key from
env vars.
"""
from __future__ import annotations

import os
from typing import Any

try:  # httpx is only needed for the live backends
    import httpx
except ImportError:  # pragma: no cover - offline/mock path does not need it
    httpx = None  # type: ignore[assignment]

from .base import Tool

# A tiny canned corpus so ``web_search`` returns something deterministic
# offline. Keys are lowercased substrings matched against the query.
_CANNED: dict[str, list[dict[str, str]]] = {
    "iceland": [
        {
            "title": "Iceland - Population",
            "url": "https://example.org/iceland",
            "snippet": "Iceland had an estimated population of about 393,600 people in 2024, "
            "making it the most sparsely populated country in Europe.",
        }
    ],
    "eiffel tower": [
        {
            "title": "Eiffel Tower - Height",
            "url": "https://example.org/eiffel-tower",
            "snippet": "The Eiffel Tower in Paris stands 330 metres tall including its "
            "antennas and was completed in 1889.",
        }
    ],
    "speed of light": [
        {
            "title": "Speed of light",
            "url": "https://example.org/speed-of-light",
            "snippet": "The speed of light in a vacuum is exactly 299,792,458 metres per second.",
        }
    ],
    "python": [
        {
            "title": "Python (programming language)",
            "url": "https://example.org/python",
            "snippet": "Python is a high-level, general-purpose programming language first "
            "released by Guido van Rossum in 1991.",
        }
    ],
}


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web and return a list of {title, url, snippet}."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, provider: str = "tavily", timeout: float = 15.0) -> None:
        if provider not in {"tavily", "serpapi", "mock"}:
            raise ValueError(f"unsupported provider {provider!r}")
        self.provider = provider
        self.timeout = timeout

    def run(self, query: str, max_results: int = 5, **_: Any) -> dict[str, Any]:
        if self.provider == "mock":
            return self._mock(query, max_results)
        if self.provider == "tavily":
            return self._tavily(query, max_results)
        return self._serpapi(query, max_results)

    def _mock(self, query: str, max_results: int) -> dict[str, Any]:
        q = query.lower()
        for key, results in _CANNED.items():
            if key in q:
                return {"results": results[:max_results]}
        return {
            "results": [
                {
                    "title": "No canned result",
                    "url": "https://example.org/",
                    "snippet": f"The offline corpus has no entry for '{query}'.",
                }
            ]
        }

    def _tavily(self, query: str, max_results: int) -> dict[str, Any]:
        key = os.getenv("TAVILY_API_KEY", "")
        if not key:
            return {"error": "TAVILY_API_KEY not set", "results": []}
        try:
            r = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "results": []}
        out = []
        for item in data.get("results", [])[:max_results]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "") or item.get("snippet", ""),
                }
            )
        return {"results": out}

    def _serpapi(self, query: str, max_results: int) -> dict[str, Any]:
        key = os.getenv("SERPAPI_API_KEY", "")
        if not key:
            return {"error": "SERPAPI_API_KEY not set", "results": []}
        try:
            r = httpx.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": key, "num": max_results},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "results": []}
        out = []
        for item in data.get("organic_results", [])[:max_results]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return {"results": out}
