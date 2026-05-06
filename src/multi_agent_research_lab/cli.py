"""Command-line entrypoint for the Multi-Agent Research Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import export_trace_json, reset_run
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab CLI", no_args_is_help=True)
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


# ---------------------------------------------------------------------------
# baseline command
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save: Annotated[bool, typer.Option("--save", help="Save output to reports/")] = False,
) -> None:
    """Run the single-agent baseline: search + one LLM call."""
    _init()
    reset_run()

    llm = LLMClient()
    search = SearchClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    system_prompt = (
        "You are a research assistant. Given a query and search results, write a clear, "
        "comprehensive answer of approximately 400-600 words. Include a References section."
    )
    sources = search.search(query, max_results=request.max_sources)
    state.sources = sources
    context = "\n\n".join(
        "[" + str(i + 1) + "] " + s.title + "\n" + s.snippet
        for i, s in enumerate(sources)
    )
    user_prompt = "Query: " + query + "\n\nSearch results:\n" + context + "\n\nWrite your answer:"
    response = llm.complete(system_prompt, user_prompt)
    state.final_answer = response.content

    console.print(Panel.fit(Markdown(state.final_answer), title="[bold green]Single-Agent Baseline"))
    if save:
        _save_output(state, "baseline")


# ---------------------------------------------------------------------------
# multi-agent command
# ---------------------------------------------------------------------------

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic", help="Run CriticAgent after writer")] = False,
    save: Annotated[bool, typer.Option("--save", help="Save output + trace to reports/")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print full state as JSON")] = False,
) -> None:
    """Run the full multi-agent workflow (Supervisor -> Researcher -> Analyst -> Writer)."""
    _init()
    run_id = reset_run()

    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(use_critic=critic)
    result = workflow.run(state)

    if json_output:
        console.print_json(result.model_dump_json(indent=2))
    else:
        _print_result(result, run_id)

    if save:
        _save_output(result, "multi_agent", run_id=run_id)


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------

@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = (
        "Research GraphRAG state-of-the-art and write a 500-word summary"
    ),
    save: Annotated[bool, typer.Option("--save", help="Write report to reports/")] = True,
) -> None:
    """Compare single-agent baseline vs multi-agent workflow."""
    _init()

    llm = LLMClient()
    search = SearchClient()

    def single_runner(q: str) -> ResearchState:
        reset_run()
        request = ResearchQuery(query=q)
        state = ResearchState(request=request)
        sources = search.search(q, max_results=request.max_sources)
        state.sources = sources
        ctx = "\n\n".join(
            "[" + str(i + 1) + "] " + s.title + "\n" + s.snippet
            for i, s in enumerate(sources)
        )
        sys_p = "You are a research assistant. Write a comprehensive answer with references."
        usr_p = "Query: " + q + "\n\nSearch results:\n" + ctx + "\n\nWrite your answer:"
        resp = llm.complete(sys_p, usr_p)
        state.final_answer = resp.content
        return state

    def multi_runner(q: str) -> ResearchState:
        reset_run()
        state = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow(llm=llm, search=search).run(state)

    console.print("[bold cyan]Running benchmark...[/]")

    state_single, metrics_single = run_benchmark("single-agent", query, single_runner)
    console.print(
        "  [green]OK[/] Single-agent: "
        + str(metrics_single.latency_seconds) + "s  "
        + "quality=" + str(metrics_single.quality_score)
    )

    state_multi, metrics_multi = run_benchmark("multi-agent", query, multi_runner)
    console.print(
        "  [green]OK[/] Multi-agent : "
        + str(metrics_multi.latency_seconds) + "s  "
        + "quality=" + str(metrics_multi.quality_score)
    )

    report_md = render_markdown_report([metrics_single, metrics_multi])
    console.print("\n")
    console.print(Markdown(report_md))

    if save:
        out_path = Path("reports") / "benchmark_report.md"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(report_md, encoding="utf-8")
        console.print("[dim]Report saved -> " + str(out_path) + "[/]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_result(state: ResearchState, run_id: str) -> None:
    route_str = " -> ".join(state.route_history) or "(none)"
    console.print("\n[dim]Run ID:[/] " + run_id)
    console.print("[dim]Route:[/]  " + route_str)
    console.print("[dim]Sources:[/] " + str(len(state.sources)))
    if state.errors:
        console.print("[yellow]Warnings:[/] " + "; ".join(state.errors))

    if state.final_answer:
        console.print(Panel.fit(Markdown(state.final_answer), title="[bold green]Final Answer"))
    else:
        console.print("[red]No final answer produced.[/]")

    table = Table(title="Agent Results", show_header=True)
    table.add_column("Agent")
    table.add_column("Words", justify="right")
    table.add_column("Cost USD", justify="right")
    for r in state.agent_results:
        words = str(len(r.content.split()))
        cost_val = r.metadata.get("cost_usd")
        cost = ("$" + format(cost_val, ".5f")) if cost_val else "-"
        table.add_row(r.agent, words, cost)
    console.print(table)


def _save_output(state: ResearchState, run_type: str, run_id: str | None = None) -> None:
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    suffix = "_" + run_id[:8] if run_id else ""

    state_path = out_dir / (run_type + "_state" + suffix + ".json")
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    trace_path = out_dir / (run_type + "_trace" + suffix + ".json")
    export_trace_json(trace_path)
    console.print("[dim]Saved -> " + str(state_path) + ", " + str(trace_path) + "[/]")


if __name__ == "__main__":
    app()
