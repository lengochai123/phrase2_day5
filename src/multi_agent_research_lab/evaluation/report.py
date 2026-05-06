"""Benchmark report rendering."""

from datetime import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a rich markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Benchmark Report",
        "",
        f"_Generated: {now}_",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "—" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.5f}"
        quality = "—" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")

    if len(metrics) >= 2:
        lines += ["", "## Comparison Analysis", ""]
        fastest = min(metrics, key=lambda m: m.latency_seconds)
        slowest = max(metrics, key=lambda m: m.latency_seconds)
        lines.append(f"- **Fastest**: `{fastest.run_name}` ({fastest.latency_seconds:.2f}s)")
        lines.append(f"- **Slowest**: `{slowest.run_name}` ({slowest.latency_seconds:.2f}s)")

        speedup = slowest.latency_seconds / fastest.latency_seconds if fastest.latency_seconds > 0 else 1.0
        lines.append(f"- **Latency ratio**: {speedup:.1f}× difference")

        scored = [m for m in metrics if m.quality_score is not None]
        if scored:
            best_quality = max(scored, key=lambda m: m.quality_score or 0)
            lines.append(f"- **Highest quality**: `{best_quality.run_name}` (score {best_quality.quality_score:.1f}/10)")

        costed = [m for m in metrics if m.estimated_cost_usd is not None]
        if costed:
            cheapest = min(costed, key=lambda m: m.estimated_cost_usd or 0)
            lines.append(f"- **Cheapest**: `{cheapest.run_name}` (${cheapest.estimated_cost_usd:.5f})")

    lines += [
        "",
        "## Interpretation",
        "",
        "| Dimension | Single-Agent | Multi-Agent |",
        "|---|---|---|",
        "| Latency | Lower (1 LLM call) | Higher (multiple agent calls) |",
        "| Cost | Lower (fewer tokens) | Higher (researcher + analyst + writer) |",
        "| Quality | Baseline | Higher (specialized roles + search) |",
        "| Traceability | Low (black box) | High (per-agent trace events) |",
        "| Failure isolation | None | Per-agent (supervisor fallback) |",
        "",
        "> **When to use multi-agent**: When answer quality and citation coverage matter more than latency/cost.",
        "> **When NOT to**: Simple lookups, tight latency budgets, or when a single well-prompted LLM is sufficient.",
        "",
    ]

    return "\n".join(lines)
