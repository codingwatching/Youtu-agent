"""call_model_input_filter implementations for RunConfig.

These filters are invoked immediately before each model call, replacing the
custom logic previously baked into UTUAgentRunner (_context_manager_preprocess,
inject context infos, _check_too_long).
"""

from __future__ import annotations

import logging
from typing import Any

from agents import TResponseInputItem
from agents.run_config import CallModelData, ModelInputData

from ..config import AgentConfig
from ..context import BaseContextManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal counter key stored on the context dict to track turns.
# We use a dunder name to avoid collisions with user keys.
# ---------------------------------------------------------------------------
_TURN_COUNTER_KEY = "__utu_turn_counter"


def utu_input_filter(data: CallModelData[Any]) -> ModelInputData:
    """Unified call_model_input_filter that performs:

    1. Turn counter injection  (current_turn / max_turns)
    2. Context-manager preprocessing (DummyContextManager, EnvContextManager, …)
    3. Token-budget enforcement  (strip tool calls when over termination_max_tokens)
    """
    ctx: dict[str, Any] | None = data.context
    if not isinstance(ctx, dict):
        return data.model_data

    _inject_turn_info(ctx)
    _apply_context_manager(data.model_data, ctx)
    _enforce_token_budget(data.model_data, ctx)

    return data.model_data


# ---------------------------------------------------------------------------
# 1. Turn counter
# ---------------------------------------------------------------------------

def _inject_turn_info(ctx: dict[str, Any]) -> None:
    """Increment and store ``current_turn`` on the context dict.

    ``max_turns`` must already be present in the context (set by SimpleAgent
    before starting the run).
    """
    counter: int = ctx.get(_TURN_COUNTER_KEY, 0) + 1
    ctx[_TURN_COUNTER_KEY] = counter
    ctx["current_turn"] = counter


# ---------------------------------------------------------------------------
# 2. Context-manager preprocessing
# ---------------------------------------------------------------------------

def _apply_context_manager(
    model_data: ModelInputData,
    ctx: dict[str, Any],
) -> None:
    cm: BaseContextManager | None = ctx.get("context_manager")
    if cm is None:
        return
    model_data.input = cm.preprocess(model_data.input, ctx)


# ---------------------------------------------------------------------------
# 3. Token-budget enforcement
# ---------------------------------------------------------------------------

def _enforce_token_budget(
    model_data: ModelInputData,
    ctx: dict[str, Any],
) -> None:
    """If the *previous* response already exceeded ``termination_max_tokens``,
    inject a user message asking the model to stop using tools and provide a
    final answer.

    The actual token count is only available *after* a model response, so we
    check the running total recorded by the framework on ``context.usage``.
    The very first call has no prior usage, so this is a no-op on turn 1.
    """
    config: AgentConfig | None = ctx.get("agent_config")
    if config is None or not config.model.termination_max_tokens:
        return

    max_tokens = config.model.termination_max_tokens
    # ``__utu_last_total_tokens`` is set by the on_llm_end hook (see hooks.py)
    last_total: int = ctx.get("__utu_last_total_tokens", 0)
    if last_total <= 0 or last_total <= max_tokens:
        return

    logger.warning(
        "Token budget exceeded (last_total=%d, max=%d). "
        "Injecting stop-tool message.",
        last_total,
        max_tokens,
    )
    stop_msg: TResponseInputItem = {
        "role": "user",
        "content": (
            "You have exceeded the token budget. "
            "Please DO NOT use ANY tools, provide your final answer immediately."
        ),
    }
    model_data.input.append(stop_msg)
