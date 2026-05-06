# Benchmark Report

_Generated: 2026-05-06 17:18:09_

## Results

| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |
|---|---:|---:|---:|---|
| single-agent-baseline | 11.75 | — | — | single LLM call, no search |

## Interpretation

| Dimension | Single-Agent | Multi-Agent |
|---|---|---|
| Latency | Lower (1 LLM call) | Higher (multiple agent calls) |
| Cost | Lower (fewer tokens) | Higher (researcher + analyst + writer) |
| Quality | Baseline | Higher (specialized roles + search) |
| Traceability | Low (black box) | High (per-agent trace events) |
| Failure isolation | None | Per-agent (supervisor fallback) |

> **When to use multi-agent**: When answer quality and citation coverage matter more than latency/cost.
> **When NOT to**: Simple lookups, tight latency budgets, or when a single well-prompted LLM is sufficient.
