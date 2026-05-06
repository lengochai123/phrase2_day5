"""LLM client abstraction - OpenAI implementation.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

# gpt-4o-mini pricing per 1M tokens (USD)
_COST_INPUT_PER_M = 0.15
_COST_OUTPUT_PER_M = 0.60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """OpenAI chat completion client with retry and token tracking."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._timeout = settings.timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with token usage and cost estimate."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self._timeout,
            )
        except openai.OpenAIError as exc:
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = (input_tokens * _COST_INPUT_PER_M + output_tokens * _COST_OUTPUT_PER_M) / 1_000_000

        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
