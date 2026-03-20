"""Concurrent rollout script for testing reasoning model (e.g. GLM-5).

Runs N tasks concurrently using SimpleAgent with base_search config,
saves chat.completions messages (with reasoning fields) to JSONL.

Usage:
    python examples/rollout/run_concurrent.py
    python examples/rollout/run_concurrent.py --model glm-5-0304 --concurrency 5 --output results.jsonl
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from agents import trace

from utu.agents import SimpleAgent
from utu.utils import AgentsUtils, ChatCompletionConverter

MOCK_QUERIES = [
    "What is the latest news about SpaceX Starship in 2026?",
    "Compare the GDP of China and USA in 2025.",
    "Who won the Nobel Prize in Physics 2025 and what was their contribution?",
    "What are the key features of Python 3.13?",
    "Search for the most popular open-source LLM frameworks in 2026.",
]


async def run_single_task(
    query: str,
    task_id: int,
    config: str,
    model: str | None = None,
) -> dict:
    """Run a single agent task and return the chat.completions messages."""
    print(f"[Task {task_id}] Starting: {query[:60]}...")
    t0 = time.time()

    try:
        async with SimpleAgent(config=config, model=model) as agent:
            trace_id = AgentsUtils.gen_trace_id()
            with trace(workflow_name="rollout", trace_id=trace_id):
                recorder = await agent.run(query, trace_id=trace_id, log_to_db=False)

        # Convert to chat.completions messages format (with reasoning fields)
        input_list = recorder.get_run_result().to_input_list()
        messages = ChatCompletionConverter.items_to_messages(input_list)

        elapsed = time.time() - t0
        print(f"[Task {task_id}] Done in {elapsed:.1f}s, {len(messages)} messages")

        return {
            "task_id": task_id,
            "query": query,
            "trace_id": trace_id,
            "messages": messages,
            "final_output": recorder.final_output,
            "elapsed": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Task {task_id}] Error after {elapsed:.1f}s: {e}")
        return {
            "task_id": task_id,
            "query": query,
            "error": str(e),
            "elapsed": round(elapsed, 2),
        }


async def main(
    config: str = "simple/base_search",
    model: str | None = None,
    concurrency: int = 5,
    output: str = "rollout_results.jsonl",
    queries: list[str] | None = None,
):
    queries = queries or MOCK_QUERIES
    print(f"Running {len(queries)} tasks with concurrency={concurrency}")
    print(f"Config: {config}, Model: {model or 'default'}")
    print(f"Output: {output}\n")

    sem = asyncio.Semaphore(concurrency)

    async def bounded_run(query: str, task_id: int) -> dict:
        async with sem:
            return await run_single_task(query, task_id, config, model)

    tasks = [bounded_run(q, i) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks)

    # Write results to JSONL
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    succeeded = sum(1 for r in results if "error" not in r)
    print(f"\n{'=' * 60}")
    print(f"Results: {succeeded}/{len(results)} succeeded")
    print(f"Output saved to: {output_path.resolve()}")

    # Check if reasoning fields are present
    reasoning_count = 0
    for r in results:
        for msg in r.get("messages", []):
            if msg.get("reasoning") or msg.get("reasoning_content"):
                reasoning_count += 1
    if reasoning_count:
        print(f"Reasoning fields found in {reasoning_count} messages")
    else:
        print("WARNING: No reasoning fields found in any messages!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent rollout for reasoning model testing")
    parser.add_argument("--config", default="simple/base_search", help="Agent config name")
    parser.add_argument("--model", default=None, help="Override model name (e.g. glm-5-0304)")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent tasks")
    parser.add_argument("--output", default="rollout_results.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    asyncio.run(main(config=args.config, model=args.model, concurrency=args.concurrency, output=args.output))
