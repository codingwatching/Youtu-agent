from typing import Literal

from agents import RunConfig, Runner
from agents.run import AgentRunner

from .react_runner import ReactRunner


def get_runner(name: Literal["openai", "react"] = "openai") -> object:
    """Get a runner class by name.

    Args:
        name: Runner name ("openai" for default, "react" for ReactRunner)

    Returns:
        Runner class (not instance)
    """
    # TODO: add a protocol for runner
    if name == "react":
        return ReactRunner
    elif name == "openai":
        return AgentRunner()
    else:
        raise ValueError(f"Unknown runner name: {name}")


__all__ = ["Runner", "RunConfig", "ReactRunner", "get_runner"]
