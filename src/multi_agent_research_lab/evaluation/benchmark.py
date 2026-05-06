"""Benchmark runner for single-agent vs multi-agent comparison."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Time a runner, extract quality and cost metrics, return both."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    total_cost = _total_cost(state)
    quality_score = _compute_quality(state)

    notes_parts = [
        "routes=" + " -> ".join(state.route_history),
        "sources=" + str(len(state.sources)),
        "errors=" + str(len(state.errors)),
    ]
    if state.errors:
        notes_parts.append("err_sample=" + state.errors[0][:60])

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(total_cost, 6) if total_cost else None,
        quality_score=quality_score,
        notes=" | ".join(notes_parts),
    )
    logger.info(
        "Benchmark %r: latency=%.2fs cost=$%.5f quality=%s",
        run_name, latency, total_cost or 0.0, quality_score,
    )
    return state, metrics


def _total_cost(state: ResearchState) -> float:
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost:
            total += float(cost)
    return total


def _compute_quality(state: ResearchState) -> float | None:
    """Heuristic quality score 0-10 based on output completeness.

    Dimensions:
    - Has final_answer (4 pts) + word count bonus (0-1 pt)
    - Has research_notes (2 pts)
    - Has analysis_notes (2 pts)
    - Citation coverage (0-1 pt)
    - Critic score overrides if present
    """
    for result in state.agent_results:
        if result.agent == AgentName.CRITIC:
            critic_score = result.metadata.get("quality_score")
            if critic_score is not None:
                return float(critic_score)

    score = 0.0
    if state.final_answer:
        score += 4.0
        word_count = len(state.final_answer.split())
        score += min(1.0, word_count / 400)
    if state.research_notes:
        score += 2.0
    if state.analysis_notes:
        score += 2.0
    if state.sources and state.final_answer:
        cited = sum(
            1 for src in state.sources
            if src.title.split(":")[0][:20].lower() in state.final_answer.lower()
        )
        score += min(1.0, cited / max(1, len(state.sources)))

    return round(min(10.0, score), 1)
