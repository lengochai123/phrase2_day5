"""Benchmark runner for single-agent vs multi-agent comparison."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, and quality for a runner function.

    Extracts:
    - Latency: wall-clock time
    - Cost: sum of cost_usd from all agent results
    - Quality: score from CriticAgent if present (else None)
    - Notes: agent call count and error count
    """
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    total_cost = sum(
        float(r.metadata.get("cost_usd") or 0)
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    )

    quality_score: float | None = None
    for result in reversed(state.agent_results):
        if result.agent == AgentName.CRITIC:
            raw = result.metadata.get("quality_score")
            if raw is not None:
                quality_score = float(raw)
            break

    citation_count = sum(1 for s in state.sources if s.url)
    notes = (
        f"{len(state.agent_results)} agent calls, "
        f"{len(state.sources)} sources, "
        f"{citation_count} cited, "
        f"{len(state.errors)} errors"
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(total_cost, 6) if total_cost > 0 else None,
        quality_score=quality_score,
        notes=notes,
    )
    return state, metrics
