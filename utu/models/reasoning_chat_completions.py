"""Reasoning-aware ChatCompletions model and strategy pattern.

Extends OpenAIChatCompletionsModel to properly handle reasoning/thinking fields
for models like GLM-5, DeepSeek, etc. Each model family has a ReasoningStrategy
that knows how to extract, inject, and configure its specific reasoning format.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from agents.models.chatcmpl_converter import Converter
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ReasoningStrategy(Protocol):
    """Adapts reasoning fields for a specific model family.

    Each model family (GLM-5, DeepSeek, Qwen, etc.) may expose reasoning/thinking
    content through different field names and require different API parameters.
    Implement this protocol to add support for a new model family.
    """

    def extract_reasoning(self, message: ChatCompletionMessage) -> str | None:
        """Extract reasoning text from a model response message."""
        ...

    def inject_reasoning(self, assistant_msg: dict, reasoning_text: str) -> None:
        """Inject reasoning text into an assistant message dict for multi-turn context."""
        ...

    def get_extra_body(self) -> dict | None:
        """Return model-specific extra_body params to merge into the API call."""
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class GLM5ReasoningStrategy:
    """GLM-5 uses ``message.reasoning`` for interleaved thinking."""

    def extract_reasoning(self, message: ChatCompletionMessage) -> str | None:
        return getattr(message, "reasoning", None) or None

    def inject_reasoning(self, assistant_msg: dict, reasoning_text: str) -> None:
        assistant_msg["reasoning"] = reasoning_text

    def get_extra_body(self) -> dict | None:
        return {"chat_template_kwargs": {"clear_thinking": False}}


class DeepSeekReasoningStrategy:
    """DeepSeek uses ``message.reasoning_content``."""

    def extract_reasoning(self, message: ChatCompletionMessage) -> str | None:
        return getattr(message, "reasoning_content", None) or None

    def inject_reasoning(self, assistant_msg: dict, reasoning_text: str) -> None:
        assistant_msg["reasoning_content"] = reasoning_text

    def get_extra_body(self) -> dict | None:
        return None


class NullReasoningStrategy:
    """No-op strategy for models without reasoning support."""

    def extract_reasoning(self, message: ChatCompletionMessage) -> str | None:
        return None

    def inject_reasoning(self, assistant_msg: dict, reasoning_text: str) -> None:
        pass

    def get_extra_body(self) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: list[tuple[str, type[ReasoningStrategy]]] = [
    ("glm", GLM5ReasoningStrategy),
    ("deepseek", DeepSeekReasoningStrategy),
]


def get_reasoning_strategy(model_name: str) -> ReasoningStrategy:
    """Resolve a ReasoningStrategy from a model name by substring match."""
    lower = model_name.lower()
    for keyword, strategy_cls in _STRATEGY_REGISTRY:
        if keyword in lower:
            return strategy_cls()
    return NullReasoningStrategy()


def register_reasoning_strategy(keyword: str, strategy_cls: type[ReasoningStrategy]) -> None:
    """Register a new reasoning strategy for a model family keyword.

    Example::

        register_reasoning_strategy("qwen", QwenReasoningStrategy)
    """
    _STRATEGY_REGISTRY.insert(0, (keyword.lower(), strategy_cls))


# ---------------------------------------------------------------------------
# Model subclass
# ---------------------------------------------------------------------------


class ReasoningChatCompletionsModel(OpenAIChatCompletionsModel):
    """ChatCompletions model with pluggable reasoning field handling.

    Intercepts two data paths:

    1. **Output (model → framework)**: After receiving a ChatCompletion, extracts
       the reasoning field via the strategy and normalizes it to
       ``message.reasoning_content`` so the upstream Converter picks it up.

    2. **Input (framework → model)**: The upstream Converter produces
       ``reasoning_content`` on assistant messages (DeepSeek path). We rewrite
       that to the strategy's field name (e.g. ``reasoning`` for GLM-5).

    3. **API params**: Merges strategy-specific ``extra_body`` (e.g. GLM-5's
       ``clear_thinking``) into ``model_settings`` before each call.
    """

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        reasoning_strategy: ReasoningStrategy | None = None,
    ) -> None:
        super().__init__(model=model, openai_client=openai_client)
        self._reasoning_strategy = reasoning_strategy or get_reasoning_strategy(model)

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        # _fetch_response(self, system_instructions, input, model_settings, ...)
        model_settings = args[2] if len(args) > 2 else kwargs.get("model_settings")

        # --- 1) Merge strategy extra_body into model_settings ---
        if model_settings is not None:
            self._merge_extra_body(model_settings)

        # --- 2) Input path: patch Converter.items_to_messages to activate the
        #     DeepSeek reasoning_content path, then rewrite field names. ---
        needs_reasoning_rewrite = not isinstance(
            self._reasoning_strategy, (DeepSeekReasoningStrategy, NullReasoningStrategy)
        )
        if needs_reasoning_rewrite:
            original_items_to_messages = Converter.items_to_messages
            patched = self._make_patched_items_to_messages(original_items_to_messages)
            Converter.items_to_messages = patched  # type: ignore[assignment]

        try:
            result = await super()._fetch_response(*args, **kwargs)
        finally:
            if needs_reasoning_rewrite:
                Converter.items_to_messages = original_items_to_messages  # type: ignore[assignment]

        # --- 3) Output path: normalize reasoning on non-streaming responses ---
        if isinstance(result, ChatCompletion):
            self._normalize_reasoning_output(result)

        return result

    def _merge_extra_body(self, model_settings: Any) -> None:
        """Deep-merge strategy extra_body into model_settings.extra_body."""
        strategy_body = self._reasoning_strategy.get_extra_body()
        if not strategy_body:
            return
        existing = model_settings.extra_body or {}
        merged = {**existing}
        for k, v in strategy_body.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        model_settings.extra_body = merged

    def _normalize_reasoning_output(self, response: ChatCompletion) -> None:
        """Extract reasoning via strategy and bridge it to upstream's expected field.

        The upstream ``Converter.message_to_output_items`` checks:
        ``hasattr(message, 'reasoning_content') and message.reasoning_content``

        For models like GLM-5 where the field is ``reasoning`` instead, we copy
        it to ``reasoning_content`` so the upstream code path works.
        """
        if not response.choices:
            return
        message = response.choices[0].message
        reasoning = self._reasoning_strategy.extract_reasoning(message)
        if reasoning and not getattr(message, "reasoning_content", None):
            message.reasoning_content = reasoning  # type: ignore[attr-defined]

    def _make_patched_items_to_messages(
        self, original_items_to_messages: Any
    ) -> classmethod:
        """Build a patched ``Converter.items_to_messages`` classmethod.

        The upstream Converter's DeepSeek reasoning path has ordering
        assumptions that break for other models (e.g. GLM-5):

        - It stores reasoning in ``pending_reasoning_content``
        - But ``flush_assistant_message()`` clears it when the next item is a
          ``response_output_message`` (which calls flush before
          ``current_assistant_msg`` exists).

        Instead of hacking the DeepSeek path, we handle reasoning entirely
        ourselves:

        1. Extract reasoning text from reasoning items.
        2. Track which ``response_output_message`` each reasoning belongs to
           (the immediately following one).
        3. Run the converter on items **stripped** of reasoning items.
        4. Post-inject reasoning into the correct assistant messages.
        """
        strategy = self._reasoning_strategy

        @classmethod  # type: ignore[misc]
        def patched_items_to_messages(cls, items, model=None, **kw):  # type: ignore[no-untyped-def]
            # --- Pass 1: extract reasoning & build stripped item list ---
            stripped_items: list = []
            pending_reasoning: str | None = None
            reasoning_map: dict[int, str] = {}

            for item in items:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    content_items = item.get("content", [])
                    texts = [
                        c["text"]
                        for c in content_items
                        if isinstance(c, dict)
                        and c.get("type") == "reasoning_text"
                        and c.get("text")
                    ]
                    if not texts:
                        summary_items = item.get("summary", [])
                        texts = [
                            s["text"]
                            for s in summary_items
                            if isinstance(s, dict) and s.get("text")
                        ]
                    if texts:
                        reasoning_text = "\n".join(texts)
                        if pending_reasoning:
                            pending_reasoning += "\n" + reasoning_text
                        else:
                            pending_reasoning = reasoning_text
                    continue

                if pending_reasoning is not None:
                    reasoning_map[len(stripped_items)] = pending_reasoning
                    pending_reasoning = None
                stripped_items.append(item)

            # --- Pass 2: run the normal converter on stripped items ---
            messages = original_items_to_messages.__func__(
                cls, stripped_items, model=model, **kw
            )

            # --- Pass 3: inject reasoning into assistant messages ---
            # Walk stripped_items counting response_output_messages to correlate
            # with output assistant messages by ordinal position.
            if reasoning_map:
                reasoning_indices = sorted(reasoning_map.keys())
                asst_count = 0
                reasoning_iter = iter(reasoning_indices)
                next_target = next(reasoning_iter, None)
                output_reasoning: dict[int, str] = {}
                for si_idx, si_item in enumerate(stripped_items):
                    is_resp_output = (
                        isinstance(si_item, dict)
                        and si_item.get("type") == "message"
                        and si_item.get("role") == "assistant"
                    )
                    if si_idx == next_target:
                        if is_resp_output:
                            output_reasoning[asst_count] = reasoning_map[si_idx]
                        next_target = next(reasoning_iter, None)
                    if is_resp_output:
                        asst_count += 1

                asst_idx = 0
                for msg in messages:
                    if msg.get("role") == "assistant":
                        if asst_idx in output_reasoning:
                            strategy.inject_reasoning(msg, output_reasoning[asst_idx])
                        asst_idx += 1

            return messages

        return patched_items_to_messages
