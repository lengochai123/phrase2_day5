"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline and save a benchmark report."""
    _init()

    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.services.storage import LocalArtifactStore

    def _run_single_agent(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient()
        response = llm.complete(
            system_prompt=(
                "You are a research assistant. Answer the query with clear, "
                "well-structured information including key findings, analysis, "
                "and a concise conclusion."
            ),
            user_prompt=q,
        )
        state.final_answer = response.content
        return state

    state, metrics = run_benchmark("single-agent-baseline", query, _run_single_agent)
    metrics = metrics.model_copy(update={"notes": "single LLM call, no search"})

    console.print(Panel.fit(state.final_answer or "", title="[bold]Single-Agent Baseline[/bold]"))

    report = render_markdown_report([metrics])
    store = LocalArtifactStore()
    path = store.write_text("baseline_report.md", report)
    console.print(f"\n[green]Report saved → {path}[/green]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    with_critic: Annotated[bool, typer.Option("--critic/--no-critic", help="Run critic agent")] = False,
) -> None:
    """Run the full multi-agent workflow and save a benchmark report."""
    _init()

    from multi_agent_research_lab.agents.critic import CriticAgent
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    from multi_agent_research_lab.services.storage import LocalArtifactStore

    def _run_multi_agent(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow()
        result = workflow.run(state)
        if with_critic and result.final_answer:
            result = CriticAgent().run(result)
        return result

    try:
        state, metrics = run_benchmark("multi-agent", query, _run_multi_agent)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(Panel.fit(state.final_answer or "No answer produced.", title="[bold]Multi-Agent Result[/bold]"))
    console.print(f"\n[dim]Agents run: {' → '.join(state.route_history)}[/dim]")
    if state.errors:
        console.print(f"[yellow]Warnings: {state.errors}[/yellow]")

    report = render_markdown_report([metrics])
    store = LocalArtifactStore()
    path = store.write_text("multi_agent_report.md", report)
    console.print(f"[green]Report saved → {path}[/green]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run both baseline and multi-agent, then produce a comparison report."""
    _init()

    from multi_agent_research_lab.agents.critic import CriticAgent
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.services.storage import LocalArtifactStore

    def _baseline(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient()
        resp = llm.complete(
            "You are a research assistant. Answer with clear, well-structured information.", q
        )
        state.final_answer = resp.content
        return state

    def _multi(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        result = MultiAgentWorkflow().run(state)
        return CriticAgent().run(result)

    console.print("[bold cyan]Running single-agent baseline...[/bold cyan]")
    _, baseline_metrics = run_benchmark("single-agent", query, _baseline)
    baseline_metrics = baseline_metrics.model_copy(update={"notes": "single LLM call"})

    console.print("[bold cyan]Running multi-agent workflow...[/bold cyan]")
    multi_state, multi_metrics = run_benchmark("multi-agent", query, _multi)
    multi_metrics = multi_metrics.model_copy(
        update={"notes": f"agents: {' → '.join(multi_state.route_history)}"}
    )

    report = render_markdown_report([baseline_metrics, multi_metrics])
    store = LocalArtifactStore()
    path = store.write_text("benchmark_report.md", report)

    console.print(Panel.fit(report, title="[bold]Benchmark Comparison[/bold]"))
    console.print(f"[green]Full report saved → {path}[/green]")


if __name__ == "__main__":
    app()
