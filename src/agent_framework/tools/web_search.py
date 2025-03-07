"""Web search tool.

Two backends:

- ``tavily``  - simple POST to https://api.tavily.com/search
- ``serpapi`` - the SerpAPI Google endpoint

Pick a provider via the constructor; both read their key from env vars.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from .base import Tool


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
        if provider not in {"tavily", "serpapi"}:
            raise ValueError(f"unsupported provider {provider!r}")
        self.provider = provider
        self.timeout = timeout

    def run(self, query: str, max_results: int = 5, **_: Any) -> dict[str, Any]:
        if self.provider == "tavily":
            return self._tavily(query, max_results)
        return self._serpapi(query, max_results)

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
