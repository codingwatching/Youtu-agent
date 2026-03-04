"""Test patched_items_to_messages for ReasoningChatCompletionsModel.

Verifies that GLM-5 reasoning items (which store thinking text in `content`
with type=reasoning_text) are correctly injected into the converted assistant
messages via the GLM5ReasoningStrategy.

Run:  python tests/models/test_reasoning_converter.py
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

from agents.models.chatcmpl_converter import Converter

from utu.models.reasoning_chat_completions import (
    GLM5ReasoningStrategy,
    ReasoningChatCompletionsModel,
)

# ---------------------------------------------------------------------------
# Simulated GLM-5 items (captured from real debug session)
# ---------------------------------------------------------------------------

REASONING_ITEM = {
    "id": "__fake_id__",
    "summary": [],
    "type": "reasoning",
    "content": [
        {
            "text": "The user wants me to use the pptx skill to create a simple PowerPoint presentation.",
            "type": "reasoning_text",
        }
    ],
    "provider_data": {
        "model": "glm-5-fp8",
        "response_id": "chatcmpl-d356dde5-f4fc-43af-b9a2-9bd99f77908a",
    },
}

RESPONSE_OUTPUT_MESSAGE = {
    "id": "__fake_id__",
    "content": [
        {
            "annotations": [],
            "text": "I'll help you create a simple PowerPoint presentation with one page.",
            "type": "output_text",
            "logprobs": [],
        }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message",
    "provider_data": {
        "model": "glm-5-fp8",
        "response_id": "chatcmpl-f43f4337-7a46-4cdc-ae15-18e97f1dd53b",
    },
}

FUNCTION_TOOL_CALL = {
    "id": "call_001",
    "type": "function_call",
    "call_id": "call_001",
    "name": "some_tool",
    "arguments": "{}",
    "status": "completed",
}

FUNCTION_TOOL_OUTPUT = {
    "id": "tool_output_001",
    "type": "function_call_output",
    "call_id": "call_001",
    "output": "tool result here",
}

USER_INPUT = {
    "role": "user",
    "content": "use the pptx skill to create a simple ppt with one page",
    "type": "message",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


@contextmanager
def patched_converter():
    """Activate the ReasoningChatCompletionsModel converter patch for GLM-5."""
    model = ReasoningChatCompletionsModel(
        model="glm-5-fp8",
        openai_client=MagicMock(),
        reasoning_strategy=GLM5ReasoningStrategy(),
    )
    original = Converter.items_to_messages
    patched = model._make_patched_items_to_messages(original)
    Converter.items_to_messages = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        Converter.items_to_messages = original


def convert(items: list) -> list:
    """Run Converter.items_to_messages with the patch active."""
    return Converter.items_to_messages(items, model="glm-5-fp8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_reasoning_injection():
    """reasoning + response_output_message -> assistant msg with 'reasoning' field."""
    with patched_converter():
        messages = convert([USER_INPUT, REASONING_ITEM, RESPONSE_OUTPUT_MESSAGE])

    print("\n=== test_basic_reasoning_injection ===")
    for i, msg in enumerate(messages):
        print(f"  msg[{i}]: role={msg.get('role')}, keys={sorted(msg.keys())}")

    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(asst_msgs) == 1, f"Expected 1 assistant msg, got {len(asst_msgs)}"
    assert "reasoning" in asst_msgs[0], (
        f"Missing 'reasoning' field. Keys: {sorted(asst_msgs[0].keys())}"
    )
    assert "pptx skill" in asst_msgs[0]["reasoning"]
    print("  PASSED")


def test_reasoning_with_tool_call():
    """reasoning + response_output_message + tool_call + tool_output -> reasoning preserved."""
    items = [
        USER_INPUT,
        REASONING_ITEM,
        RESPONSE_OUTPUT_MESSAGE,
        FUNCTION_TOOL_CALL,
        FUNCTION_TOOL_OUTPUT,
    ]
    with patched_converter():
        messages = convert(items)

    print("\n=== test_reasoning_with_tool_call ===")
    for i, msg in enumerate(messages):
        print(f"  msg[{i}]: role={msg.get('role')}, keys={sorted(msg.keys())}")

    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(asst_msgs) == 1, f"Expected 1 assistant msg, got {len(asst_msgs)}"
    assert "reasoning" in asst_msgs[0], "Missing 'reasoning' in assistant msg"
    print("  PASSED")


def test_no_reasoning_no_injection():
    """response_output_message without preceding reasoning -> no reasoning field."""
    with patched_converter():
        messages = convert([USER_INPUT, RESPONSE_OUTPUT_MESSAGE])

    print("\n=== test_no_reasoning_no_injection ===")
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(asst_msgs) == 1
    assert "reasoning" not in asst_msgs[0], "Should NOT have reasoning field"
    print("  PASSED")


def test_multi_turn_selective_reasoning():
    """Multi-turn: only the turn with reasoning gets the field injected."""
    second_user = {"role": "user", "content": "thanks", "type": "message"}
    second_response = {
        **RESPONSE_OUTPUT_MESSAGE,
        "content": [
            {"annotations": [], "text": "You're welcome!", "type": "output_text", "logprobs": []},
        ],
    }
    items = [
        USER_INPUT,
        REASONING_ITEM,
        RESPONSE_OUTPUT_MESSAGE,
        second_user,
        second_response,  # no reasoning before this one
    ]
    with patched_converter():
        messages = convert(items)

    print("\n=== test_multi_turn_selective_reasoning ===")
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(asst_msgs) == 2, f"Expected 2 assistant msgs, got {len(asst_msgs)}"
    assert "reasoning" in asst_msgs[0], "First assistant msg should have reasoning"
    assert "reasoning" not in asst_msgs[1], "Second assistant msg should NOT have reasoning"
    print("  PASSED")


def test_consecutive_reasoning_items():
    """Multiple consecutive reasoning items are merged into one."""
    reasoning2 = {
        **REASONING_ITEM,
        "content": [
            {"text": "I also need to check the template.", "type": "reasoning_text"},
        ],
    }
    items = [USER_INPUT, REASONING_ITEM, reasoning2, RESPONSE_OUTPUT_MESSAGE]
    with patched_converter():
        messages = convert(items)

    print("\n=== test_consecutive_reasoning_items ===")
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(asst_msgs) == 1
    reasoning = asst_msgs[0].get("reasoning", "")
    assert "pptx skill" in reasoning, f"First reasoning missing: {reasoning}"
    assert "template" in reasoning, f"Second reasoning missing: {reasoning}"
    print("  PASSED")


if __name__ == "__main__":
    test_basic_reasoning_injection()
    test_reasoning_with_tool_call()
    test_no_reasoning_no_injection()
    test_multi_turn_selective_reasoning()
    test_consecutive_reasoning_items()
    print("\n=== ALL TESTS PASSED ===")
