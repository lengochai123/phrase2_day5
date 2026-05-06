"""Multi-agent workflow orchestration.

Implements a supervisor-loop pattern:
  Supervisor → (researcher | analyst | writer | done) → repeat

No external graph framework required; the loop is explicit and traceable.
"""

import logging

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the supervisor-loop multi-agent graph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._agents = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }
        self._settings = get_settings()

    def build(self) -> "MultiAgentWorkflow":
        """Return self — the loop IS the graph in this implementation."""
        return self

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the supervisor loop until done or max iterations reached."""
        max_iter = self._settings.max_iterations

        with trace_span("multi_agent_workflow", {"query": state.request.query}):
            while state.iteration < max_iter:
                with trace_span("supervisor_step", {"iteration": state.iteration}):
                    state = self._supervisor.run(state)

                route = state.route_history[-1] if state.route_history else "done"

                if route == "done":
                    logger.info("Workflow complete after %d iterations", state.iteration)
                    break

                agent = self._agents.get(route)
                if agent is None:
                    msg = f"Unknown route '{route}' returned by supervisor"
                    logger.error(msg)
                    state.errors.append(msg)
                    break

                with trace_span(f"{route}_step", {"iteration": state.iteration}):
                    state = agent.run(state)
            else:
                logger.warning("Workflow hit max_iterations=%d without finishing", max_iter)
                state.errors.append(f"Stopped at max_iterations={max_iter}")

        return state
