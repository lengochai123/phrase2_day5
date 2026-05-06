# Benchmark Report

_Generated: 2026-05-06 17:56:39_

## Results

| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |
|---|---:|---:|---:|---|
| multi-agent | 46.60 | 0.00159 | — | 3 agent calls, 5 sources, 5 cited, 0 errors |

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
