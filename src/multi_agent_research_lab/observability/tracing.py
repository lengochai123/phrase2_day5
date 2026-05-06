"""Tracing hooks -- JSON-based by default, optional LangSmith integration.

Usage:
    with trace_span("my_span", {"key": "value"}) as span:
        span["result"] = "ok"
    # span["duration_seconds"] is set on exit

To enable LangSmith, set LANGSMITH_API_KEY in .env.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

_current_run_spans: list[dict[str, Any]] = []
_run_id: str = str(uuid.uuid4())


def reset_run() -> str:
    """Reset spans for a new run and return the new run_id."""
    global _current_run_spans, _run_id
    _current_run_spans = []
    _run_id = str(uuid.uuid4())
    return _run_id


def get_run_spans() -> list[dict[str, Any]]:
    """Return all spans recorded in the current run."""
    return list(_current_run_spans)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Record a timed span. Augments with LangSmith if configured."""
    started = perf_counter()
    ts = datetime.now(timezone.utc).isoformat()
    span: dict[str, Any] = {
        "span_id": str(uuid.uuid4())[:8],
        "run_id": _run_id,
        "name": name,
        "started_at": ts,
        "attributes": dict(attributes or {}),
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        _current_run_spans.append(span)
        logger.debug(
            "SPAN %-40s  %.3fs  [%s]",
            name,
            span["duration_seconds"],
            span["status"],
        )


def export_trace_json(path: Path | None = None) -> str:
    """Serialise all spans to JSON and optionally write to a file."""
    payload = {"run_id": _run_id, "spans": _current_run_spans}
    text = json.dumps(payload, indent=2, default=str)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info("Trace exported to %s", path)
    return text
