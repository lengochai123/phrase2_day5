"""Tracing hooks -- JSON-based by default, LangSmith integration when API key set.

Usage:
    with trace_span("my_span", {"key": "value"}) as span:
        span["result"] = "ok"

To enable LangSmith, set in .env:
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=multi-agent-research-lab
"""

from __future__ import annotations

import json
import logging
import os
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

# ---------------------------------------------------------------------------
# LangSmith client (lazy-init)
# ---------------------------------------------------------------------------

_ls_client: Any = None
_ls_project: str = ""
_ls_enabled: bool = False


def _init_langsmith() -> None:
    global _ls_client, _ls_project, _ls_enabled
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        return
    try:
        from langsmith.client import Client
        _ls_client = Client(api_key=api_key)
        _ls_project = os.getenv("LANGSMITH_PROJECT", "multi-agent-research-lab")
        _ls_enabled = True
        logger.info("LangSmith tracing enabled -> project=%r", _ls_project)
    except Exception as exc:
        logger.warning("LangSmith init failed: %s", exc)


_init_langsmith()


def reset_run() -> str:
    """Reset spans for a new run and return the new run_id."""
    global _current_run_spans, _run_id
    _current_run_spans = []
    _run_id = str(uuid.uuid4())
    return _run_id


def get_run_spans() -> list[dict[str, Any]]:
    return list(_current_run_spans)


# ---------------------------------------------------------------------------
# LangSmith helpers
# ---------------------------------------------------------------------------

def _ls_post(span: dict[str, Any]) -> str | None:
    """Post a completed span to LangSmith and return its URL."""
    if not _ls_enabled or _ls_client is None:
        return None
    try:
        run_id = str(uuid.uuid4())
        name = span["name"]
        started_at = datetime.fromisoformat(span["started_at"])
        ended_at_ts = started_at.timestamp() + (span.get("duration_seconds") or 0)
        ended_at = datetime.fromtimestamp(ended_at_ts, tz=timezone.utc)

        run_type = "chain"
        if "supervisor" in name:
            run_type = "chain"
        elif "researcher" in name or "analyst" in name or "writer" in name:
            run_type = "llm"
        elif "workflow" in name:
            run_type = "chain"

        inputs = dict(span.get("attributes", {}))
        outputs = {k: v for k, v in span.items()
                   if k not in {"span_id", "run_id", "name", "started_at",
                                "attributes", "duration_seconds", "status", "error"}}
        if span.get("status") == "error":
            outputs["error"] = span.get("error")

        _ls_client.create_run(
            id=run_id,
            name=name,
            run_type=run_type,
            inputs=inputs,
            outputs=outputs,
            start_time=started_at,
            end_time=ended_at,
            error=span.get("error"),
            project_name=_ls_project,
            extra={"metadata": {"local_span_id": span["span_id"],
                                "local_run_id": span["run_id"]}},
        )
        url = _ls_client.get_run_url(run_id=run_id, project_name=_ls_project)
        return url
    except Exception as exc:
        logger.debug("LangSmith post failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main context manager
# ---------------------------------------------------------------------------

@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Record a timed span. Sends to LangSmith if LANGSMITH_API_KEY is set."""
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

        ls_url = _ls_post(span)
        if ls_url:
            span["langsmith_url"] = ls_url
            logger.info("SPAN %-35s %.3fs -> %s", name, span["duration_seconds"], ls_url)
        else:
            logger.debug("SPAN %-35s %.3fs [%s]", name, span["duration_seconds"], span["status"])


def export_trace_json(path: Path | None = None) -> str:
    """Serialise all spans to JSON and optionally write to a file."""
    payload = {"run_id": _run_id, "spans": _current_run_spans}
    if _ls_enabled:
        payload["langsmith_project"] = _ls_project
        urls = [s.get("langsmith_url") for s in _current_run_spans if s.get("langsmith_url")]
        if urls:
            payload["langsmith_urls"] = urls
    text = json.dumps(payload, indent=2, default=str)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info("Trace exported to %s", path)
    return text
