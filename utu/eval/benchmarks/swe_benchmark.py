"""SWE-bench benchmark — per-instance Docker image lifecycle.

Each SWE-bench instance requires its own Docker image (pre-installed with the
target repo at the correct commit). This benchmark overrides ``rollout_one``
to create a fresh agent + environment per instance, inject the image name,
run the agent, extract the git diff patch, and tear down the sandbox.
"""

import json
import time

from ...agents import get_agent
from ...utils import AgentsUtils, get_logger
from ..data import EvaluationSample
from .base_benchmark import BaseBenchmark

logger = get_logger(__name__, "INFO")


class SWEBenchmark(BaseBenchmark):
    """Benchmark runner for SWE-bench with per-instance environment lifecycle."""

    async def rollout_one(self, sample: EvaluationSample) -> EvaluationSample:
        """Run a single SWE-bench instance.

        1. Deep-copy the agent config so concurrent instances don't interfere.
        2. Override the Docker image and workspace from sample metadata.
        3. Build agent → run task → extract patch via ``git diff``.
        4. Always clean up the AGS sandbox.
        """
        # -- per-instance config --
        agent_config = self.config.agent.model_copy(deep=True)
        meta = sample.meta or {}
        agent_config.env.config["image"] = meta.get("image_name", agent_config.env.config.get("image", ""))
        agent_config.env.config["workspace"] = "/testbed"

        agent = get_agent(agent_config)
        trace_id = AgentsUtils.gen_trace_id()

        if hasattr(agent, "build"):
            await agent.build(trace_id=trace_id)

        patch = ""
        start_time = time.time()
        try:
            result = await agent.run(sample.augmented_question, trace_id=trace_id)

            # Extract the patch before tearing down the sandbox.
            # Write to file + read_file to avoid pexpect pager/buffer issues
            # (git diff through the bash tool can trigger pager hangs).
            if hasattr(agent, "env") and agent.env is not None:
                try:
                    await agent.env._run_bash(
                        "cd /testbed && git add -A && git diff --cached > /tmp/model.patch"
                    )
                    patch = await agent.env._read_file("/tmp/model.patch")
                except Exception as e:
                    logger.warning("Failed to extract patch for %s: %s", meta.get("instance_id", "?"), e)
        except Exception:
            raise
        finally:
            end_time = time.time()
            # Always destroy the sandbox to avoid resource leaks.
            try:
                if hasattr(agent, "cleanup"):
                    await agent.cleanup()
            except Exception as e:
                logger.warning("Error during agent cleanup for %s: %s", meta.get("instance_id", "?"), e)

        sample.update(
            trace_id=trace_id,
            response=result.final_output if result else "",
            extracted_final_answer=patch,
            time_cost=end_time - start_time,
            trajectories=json.dumps(result.trajectories, ensure_ascii=False) if result else "",
            stage="rollout",
        )
        self.dataset.save(sample)
        return sample
