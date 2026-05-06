"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_MOCK_KB: list[dict[str, str]] = [
    {
        "title": "Building Effective Agents -- Anthropic Engineering Blog",
        "url": "https://www.anthropic.com/engineering/building-effective-agents",
        "snippet": (
            "Effective agents combine tool use, memory, and planning. "
            "The key design principle is keeping each agent's scope narrow and well-defined, "
            "then composing them through a reliable orchestration layer."
        ),
        "tags": "agent multi-agent orchestration anthropic",
    },
    {
        "title": "GraphRAG: Structured Knowledge for LLM Reasoning",
        "url": "https://arxiv.org/abs/2404.16130",
        "snippet": (
            "GraphRAG augments retrieval-augmented generation with an entity-relationship graph, "
            "enabling multi-hop reasoning across documents. "
            "It achieves state-of-the-art results on knowledge-intensive benchmarks."
        ),
        "tags": "graphrag rag knowledge graph retrieval llm",
    },
    {
        "title": "LangGraph: Stateful Multi-Actor Applications with LLMs",
        "url": "https://langchain-ai.github.io/langgraph/concepts/",
        "snippet": (
            "LangGraph models agent workflows as directed graphs where nodes are Python functions "
            "and edges encode conditional routing. It provides built-in support for cycles, "
            "checkpointing, and human-in-the-loop interrupts."
        ),
        "tags": "langgraph workflow graph agent state",
    },
    {
        "title": "ReAct: Synergising Reasoning and Acting in Language Models",
        "url": "https://arxiv.org/abs/2210.03629",
        "snippet": (
            "ReAct interleaves chain-of-thought reasoning with action execution. "
            "This significantly reduces hallucination and improves factual accuracy on HotpotQA and Fever."
        ),
        "tags": "react reasoning acting tool use chain-of-thought",
    },
    {
        "title": "Benchmarking Multi-Agent vs Single-Agent LLM Systems",
        "url": "https://arxiv.org/abs/2308.11432",
        "snippet": (
            "Across 12 complex tasks, multi-agent systems scored 28% higher on human-evaluation rubrics "
            "compared to single-agent baselines, at a latency cost of ~22% and 15% higher token usage."
        ),
        "tags": "benchmark multi-agent single-agent latency quality cost",
    },
    {
        "title": "Supervisor-Worker Patterns in Agentic AI",
        "url": "https://docs.anthropic.com/en/docs/agents",
        "snippet": (
            "A Supervisor agent coordinates specialised workers by routing queries to the most capable "
            "sub-agent, aggregating results, and enforcing guardrails such as max iterations and timeout."
        ),
        "tags": "supervisor worker routing guardrail orchestration",
    },
    {
        "title": "OpenAI Agents SDK: Orchestration and Handoffs",
        "url": "https://developers.openai.com/api/docs/guides/agents/orchestration",
        "snippet": (
            "The OpenAI Agents SDK introduces handoffs -- explicit transfer of control between agents -- "
            "and guardrails -- validators that intercept inputs and outputs before forwarding them."
        ),
        "tags": "openai agents sdk handoff guardrail",
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs",
        "url": "https://arxiv.org/abs/2201.11903",
        "snippet": (
            "Chain-of-thought prompting provides step-by-step reasoning examples in the prompt. "
            "Models prompted with CoT outperform standard prompting by 20-50% on arithmetic tasks."
        ),
        "tags": "chain-of-thought prompting reasoning llm",
    },
]


def _score(doc: dict[str, str], query: str) -> int:
    query_words = set(query.lower().split())
    text = (doc["title"] + " " + doc["snippet"] + " " + doc["tags"]).lower()
    return sum(1 for w in query_words if w in text)


class SearchClient:
    """Provider-agnostic search client -- Tavily if key present, mock otherwise."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._tavily = self._build_tavily()

    def _build_tavily(self) -> object | None:
        if not self._settings.tavily_api_key:
            logger.info("TAVILY_API_KEY not set -- using mock search.")
            return None
        try:
            from tavily import TavilyClient
            return TavilyClient(api_key=self._settings.tavily_api_key)
        except ImportError:
            logger.warning("tavily-python not installed -- using mock search.")
            return None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self._tavily is not None:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            results = self._tavily.search(query=query, max_results=max_results)
            docs: list[SourceDocument] = []
            for r in results.get("results", []):
                docs.append(SourceDocument(
                    title=r.get("title", "Untitled"),
                    url=r.get("url"),
                    snippet=r.get("content", ""),
                    metadata={"score": r.get("score", 0.0)},
                ))
            logger.info("Tavily returned %d results for query: %r", len(docs), query)
            return docs
        except Exception as exc:
            logger.error("Tavily search failed: %s -- falling back to mock.", exc)
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        scored = sorted(_MOCK_KB, key=lambda d: _score(d, query), reverse=True)
        top = scored[:max_results]
        docs = [
            SourceDocument(
                title=d["title"],
                url=d["url"],
                snippet=d["snippet"],
                metadata={"source": "mock", "relevance_score": _score(d, query)},
            )
            for d in top
        ]
        logger.info("Mock search returned %d results for query: %r", len(docs), query)
        return docs
