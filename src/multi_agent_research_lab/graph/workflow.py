"""Multi-agent workflow -- LangGraph with simple state-machine fallback."""

from __future__ import annotations

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State-machine fallback
# ---------------------------------------------------------------------------

def _run_state_machine(
    state: ResearchState,
    supervisor: SupervisorAgent,
    researcher: ResearcherAgent,
    analyst: AnalystAgent,
    writer: WriterAgent,
    critic: CriticAgent,
    max_iterations: int,
) -> ResearchState:
    """Simple Python loop -- no external dependencies required."""
    logger.info("Running state-machine workflow.")
    for _ in range(max_iterations):
        state = supervisor.run(state)
        last_route = state.route_history[-1] if state.route_history else "done"
        if last_route == "done":
            break
        elif last_route == "researcher":
            state = researcher.run(state)
        elif last_route == "analyst":
            state = analyst.run(state)
        elif last_route == "writer":
            state = writer.run(state)
        elif last_route == "critic":
            state = critic.run(state)
        else:
            logger.warning("Unknown route %r -- stopping.", last_route)
            break
    return state


# ---------------------------------------------------------------------------
# LangGraph builder
# ---------------------------------------------------------------------------

def _try_build_langgraph(
    supervisor: SupervisorAgent,
    researcher: ResearcherAgent,
    analyst: AnalystAgent,
    writer: WriterAgent,
    critic: CriticAgent,
) -> object | None:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        logger.warning("langgraph not available -- using state-machine fallback.")
        return None

    def _dump(s: ResearchState) -> dict[str, Any]:
        return s.model_dump()

    def _load(d: dict[str, Any]) -> ResearchState:
        return ResearchState.model_validate(d)

    def node_supervisor(d: dict[str, Any]) -> dict[str, Any]:
        return _dump(supervisor.run(_load(d)))

    def node_researcher(d: dict[str, Any]) -> dict[str, Any]:
        return _dump(researcher.run(_load(d)))

    def node_analyst(d: dict[str, Any]) -> dict[str, Any]:
        return _dump(analyst.run(_load(d)))

    def node_writer(d: dict[str, Any]) -> dict[str, Any]:
        return _dump(writer.run(_load(d)))

    def node_critic(d: dict[str, Any]) -> dict[str, Any]:
        return _dump(critic.run(_load(d)))

    def route_after_supervisor(d: dict[str, Any]) -> str:
        history: list[str] = d.get("route_history", [])
        last = history[-1] if history else "researcher"
        return last if last in {"researcher", "analyst", "writer", "critic"} else END

    graph = StateGraph(dict)
    graph.add_node("supervisor", node_supervisor)
    graph.add_node("researcher", node_researcher)
    graph.add_node("analyst", node_analyst)
    graph.add_node("writer", node_writer)
    graph.add_node("critic", node_critic)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"researcher": "researcher", "analyst": "analyst",
         "writer": "writer", "critic": "critic", END: END},
    )
    for worker in ("researcher", "analyst", "writer", "critic"):
        graph.add_edge(worker, "supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph using LangGraph (or fallback)."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        use_critic: bool = False,
    ) -> None:
        llm = llm or LLMClient()
        search = search or SearchClient()
        self._supervisor = SupervisorAgent(llm=llm)
        self._researcher = ResearcherAgent(llm=llm, search=search)
        self._analyst = AnalystAgent(llm=llm)
        self._writer = WriterAgent(llm=llm)
        self._critic = CriticAgent(llm=llm)
        self._use_critic = use_critic
        self._settings = get_settings()
        self._graph = self.build()

    def build(self) -> object | None:
        """Create a LangGraph graph (or return None for state-machine mode)."""
        return _try_build_langgraph(
            self._supervisor, self._researcher,
            self._analyst, self._writer, self._critic,
        )

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return the final state."""
        with trace_span("workflow.run", {"query": state.request.query}) as span:
            if self._graph is not None:
                state = self._run_langgraph(state)
            else:
                state = _run_state_machine(
                    state, self._supervisor, self._researcher,
                    self._analyst, self._writer, self._critic,
                    self._settings.max_iterations,
                )
            span["route_history"] = state.route_history
            span["num_errors"] = len(state.errors)
        logger.info(
            "Workflow complete. Routes: %s | Errors: %d",
            " -> ".join(state.route_history), len(state.errors),
        )
        return state

    def _run_langgraph(self, state: ResearchState) -> ResearchState:
        try:
            result: dict[str, Any] = self._graph.invoke(state.model_dump())
            return ResearchState.model_validate(result)
        except Exception as exc:
            logger.error("LangGraph execution failed: %s -- using state machine.", exc)
            return _run_state_machine(
                state, self._supervisor, self._researcher,
                self._analyst, self._writer, self._critic,
                self._settings.max_iterations,
            )
