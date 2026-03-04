import asyncio

from agents import StopAtTools, function_tool

from utu.agents import SimpleAgent
from utu.utils import PrintUtils


@function_tool
def gen_report(data: str) -> str:
    """Generate a report based on the provided data."""
    return f"Report generated with data: {data}"


agent = SimpleAgent(
    name="sample StopAtTools",
    instructions="You are a helpful assistant.",
    tools=[gen_report],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["gen_report"]),
)


async def main():
    while True:
        user_input = await PrintUtils.async_print_input("> ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            break
        if not user_input.strip():
            continue
        recorder = await agent.chat_streamed(user_input)
    await agent.cleanup()
    print(f"recorder: {recorder.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
