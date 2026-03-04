import asyncio

from utu.agents import SimpleAgent


async def main():
    async with SimpleAgent(name="TestsAgent", instructions="You are a helpful assistant.", toolkits=["bash"]) as agent:
        # get toolkit from agent
        bash_toolkit = agent._toolkits["bash"]
        # get attr:
        print(bash_toolkit.timeout)
        # call tool:
        result = await bash_toolkit.run_bash("curl https://httpbin.org/get")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
