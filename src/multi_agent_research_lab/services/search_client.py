"""Search client abstraction for ResearcherAgent.

Uses Tavily if TAVILY_API_KEY is configured, otherwise falls back to a realistic mock.
"""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily and mock backends."""

    def __init__(self) -> None:
        self._tavily_key = get_settings().tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self._tavily_key:
            try:
                return self._tavily_search(query, max_results)
            except Exception as exc:
                logger.warning("Tavily search failed, falling back to mock: %s", exc)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        from tavily import TavilyClient  # type: ignore[import]

        client = TavilyClient(api_key=self._tavily_key)
        results = client.search(query, max_results=max_results)
        return [
            SourceDocument(
                title=r.get("title", "Untitled"),
                url=r.get("url"),
                snippet=r.get("content", ""),
                metadata={"source": "tavily", "score": r.get("score")},
            )
            for r in results.get("results", [])
        ]

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return realistic mock documents when no search API is available."""
        short_query = query[:60]
        mock_sources = [
            SourceDocument(
                title=f"Survey: {short_query} — State of the Art 2024",
                url=f"https://arxiv.org/abs/2024.00001",
                snippet=(
                    f"This comprehensive survey covers {query}. "
                    "Key contributions include a unified taxonomy, comparative benchmarks "
                    "across 15 baselines, and analysis of open challenges. "
                    "Results show 23% improvement over prior SOTA on standard benchmarks."
                ),
                metadata={"source": "mock", "year": 2024, "type": "survey"},
            ),
            SourceDocument(
                title=f"Empirical Analysis of {short_query}",
                url=f"https://arxiv.org/abs/2024.00002",
                snippet=(
                    f"We conduct an empirical study of {query} across diverse datasets. "
                    "Findings reveal that scaling model size yields diminishing returns beyond 7B parameters, "
                    "while data quality improvements yield consistent gains. "
                    "We release code and datasets for reproducibility."
                ),
                metadata={"source": "mock", "year": 2024, "type": "empirical"},
            ),
            SourceDocument(
                title=f"Practical Guide to {short_query} in Production",
                url="https://engineering.example.com/blog/practical-guide",
                snippet=(
                    f"Engineering teams deploying {query} face common challenges: "
                    "latency constraints (p99 < 200ms), cost optimization ($0.002/query target), "
                    "and hallucination mitigation. This post details battle-tested solutions "
                    "from our production systems serving 10M+ requests/day."
                ),
                metadata={"source": "mock", "year": 2024, "type": "engineering-blog"},
            ),
            SourceDocument(
                title=f"Limitations and Failure Modes in {short_query}",
                url="https://proceedings.neurips.cc/paper/2024/mock",
                snippet=(
                    f"Despite progress, {query} systems exhibit critical failure modes: "
                    "distribution shift degrades performance by up to 40%, adversarial inputs "
                    "bypass safety filters in 12% of cases, and computational costs scale "
                    "quadratically with context length. We propose mitigation strategies."
                ),
                metadata={"source": "mock", "year": 2024, "type": "analysis"},
            ),
            SourceDocument(
                title=f"Future Directions: {short_query} and Beyond",
                url="https://distill.pub/2024/mock-future",
                snippet=(
                    f"We identify five promising research directions for {query}: "
                    "(1) efficient inference via sparse attention, "
                    "(2) continual learning without catastrophic forgetting, "
                    "(3) interpretability tooling, "
                    "(4) multi-modal integration, and "
                    "(5) alignment with human preferences at scale."
                ),
                metadata={"source": "mock", "year": 2024, "type": "perspective"},
            ),
        ]
        return mock_sources[:max_results]
