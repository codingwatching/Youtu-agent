"""SWE-ReX + AGS remote coding environment.

Provides a sandboxed coding environment using SWE-ReX runtime backed by
Tencent AGS (or a direct remote connection). Tools (bash, read_file,
write_file, edit_file) are exposed via ``get_tools()`` so the agent layer
needs no knowledge of the underlying deployment.
"""

import json
import logging

from agents import FunctionTool, RunContextWrapper, TContext, Tool

from .base_env import BaseEnv

logger = logging.getLogger(__name__)

# Truncation limit for tool output returned to the agent.
_MAX_OUTPUT_CHARS = 50_000


class SWERexEnv(BaseEnv):
    """SWE-ReX + AGS remote coding environment.

    The environment is fully self-contained: it starts an AGS sandbox (or
    connects to an existing SWE-ReX server), creates a persistent bash
    session, and exposes coding tools via ``get_tools()``.

    Config keys (all optional unless noted):
        deployment_type: "ags" (default) or "remote"

        # --- AGS deployment (deployment_type="ags") ---
        secret_id / secret_key: Tencent Cloud credentials
        region, domain, tool_id, image, image_registry_type, role_arn
        cpu, memory, timeout: container resources
        port: SWE-ReX server port inside the container (default 8000)
        startup_timeout, runtime_timeout: deployment timeouts
        mount_name, mount_image, mount_image_registry_type,
        mount_path, image_subpath, mount_readonly: storage mount for SWE-ReX binary

        # --- Remote deployment (deployment_type="remote") ---
        host: hostname of existing SWE-ReX server
        auth_token: optional auth token

        # --- Session ---
        workspace: working directory for the agent (default "/")
        bash_timeout: per-command timeout in seconds (default 120)
        session_name: session identifier (default "default")
        post_startup_commands: list of commands to run after session creation
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._deployment = None
        self._runtime = None
        self._session_name: str = self._config.get("session_name", "default")
        self._workspace: str = self._config.get("workspace", "/")
        self._bash_timeout: float = float(self._config.get("bash_timeout", 120))
        self._post_startup_commands: list[str] = self._config.get("post_startup_commands", [])
        self._tools_cache: list[Tool] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def build(self):
        """Start the deployment, connect to the runtime, and create a bash session."""
        deployment_type = self._config.get("deployment_type", "ags")
        if deployment_type == "remote":
            await self._build_remote()
        else:
            await self._build_ags()

        self._runtime = self._deployment.runtime

        # Create a persistent bash session.
        from swerex.runtime.abstract import CreateBashSessionRequest

        resp = await self._runtime.create_session(
            CreateBashSessionRequest(session=self._session_name)
        )
        logger.info("SWE-ReX bash session created: %s (output=%s)", self._session_name, resp.output[:200])

        # Run post-startup commands (cd to workspace, env setup, etc.).
        startup_cmds = []
        if self._workspace and self._workspace != "/":
            startup_cmds.append(f"cd {self._workspace}")
        startup_cmds.extend(self._post_startup_commands)

        for cmd in startup_cmds:
            output = await self._run_bash(cmd)
            logger.info("Post-startup command [%s]: %s", cmd, output[:200])

    async def _build_ags(self):
        """Start AGS deployment."""
        from swerex.deployment.ags import TencentAGSDeployment

        ags_kwargs = {}
        # Map config keys to TencentAGSDeployment kwargs.
        _AGS_KEYS = [
            "secret_id", "secret_key", "http_endpoint", "skip_ssl_verify",
            "region", "domain", "tool_id",
            "image", "image_registry_type", "role_arn",
            "cpu", "memory", "timeout", "port",
            "startup_timeout", "runtime_timeout",
            "mount_name", "mount_image", "mount_image_registry_type",
            "mount_path", "image_subpath", "mount_readonly",
        ]
        for key in _AGS_KEYS:
            if key in self._config:
                ags_kwargs[key] = self._config[key]

        logger.info("Starting AGS deployment with image=%s", ags_kwargs.get("image", "N/A"))
        self._deployment = TencentAGSDeployment(**ags_kwargs)
        await self._deployment.start()
        logger.info("AGS deployment started")

    async def _build_remote(self):
        """Connect to an existing SWE-ReX server."""
        from swerex.deployment.remote import RemoteDeployment

        host = self._config.get("host", "localhost")
        port = int(self._config.get("port", 8000))
        auth_token = self._config.get("auth_token", "")
        timeout = float(self._config.get("runtime_timeout", 600.0))

        logger.info("Connecting to remote SWE-ReX at %s:%s", host, port)
        self._deployment = RemoteDeployment(
            host=host, port=port, auth_token=auth_token, timeout=timeout,
        )
        await self._deployment.start()
        logger.info("Remote SWE-ReX deployment connected")

    async def cleanup(self):
        """Stop the deployment and release resources."""
        if self._deployment is not None:
            try:
                await self._deployment.stop()
                logger.info("SWE-ReX deployment stopped")
            except Exception as e:
                logger.warning("Error stopping SWE-ReX deployment: %s", e)
        self._deployment = None
        self._runtime = None
        self._tools_cache = None

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def get_extra_sp(self) -> str:
        return ""

    def get_state(self) -> str:
        return ""

    async def get_tools(self) -> list[Tool]:
        if self._tools_cache is not None:
            return self._tools_cache

        self._tools_cache = [
            self._make_bash_tool(),
            self._make_read_file_tool(),
            self._make_write_file_tool(),
            self._make_edit_file_tool(),
        ]
        return self._tools_cache

    # ------------------------------------------------------------------
    # Tool factories
    # ------------------------------------------------------------------

    def _make_bash_tool(self) -> FunctionTool:
        async def run_bash(ctx: RunContextWrapper[TContext], input_json: str) -> str:
            params = _parse_json(input_json)
            command = params.get("command", "")
            if not command:
                return "Error: command is required"
            timeout = float(params.get("timeout", self._bash_timeout))
            return await self._run_bash(command, timeout=timeout)

        return FunctionTool(
            name="bash",
            description=(
                "Run a bash command in a persistent shell session. "
                "Working directory and environment variables persist across calls."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."},
                    "timeout": {
                        "type": "number",
                        "description": f"Timeout in seconds (default {self._bash_timeout}).",
                    },
                },
                "required": ["command"],
            },
            on_invoke_tool=run_bash,
        )

    def _make_read_file_tool(self) -> FunctionTool:
        async def read_file(ctx: RunContextWrapper[TContext], input_json: str) -> str:
            params = _parse_json(input_json)
            path = params.get("path", "")
            if not path:
                return "Error: path is required"
            return await self._read_file(path)

        return FunctionTool(
            name="read_file",
            description="Read the contents of a file at the given path.",
            params_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to read."},
                },
                "required": ["path"],
            },
            on_invoke_tool=read_file,
        )

    def _make_write_file_tool(self) -> FunctionTool:
        async def write_file(ctx: RunContextWrapper[TContext], input_json: str) -> str:
            params = _parse_json(input_json)
            path = params.get("path", "")
            content = params.get("content", "")
            if not path:
                return "Error: path is required"
            return await self._write_file(path, content)

        return FunctionTool(
            name="write_file",
            description=(
                "Write content to a file at the given path. "
                "Creates the file if it does not exist, overwrites if it does."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to write."},
                    "content": {"type": "string", "description": "The content to write to the file."},
                },
                "required": ["path", "content"],
            },
            on_invoke_tool=write_file,
        )

    def _make_edit_file_tool(self) -> FunctionTool:
        async def edit_file(ctx: RunContextWrapper[TContext], input_json: str) -> str:
            params = _parse_json(input_json)
            path = params.get("path", "")
            old_str = params.get("old_str", "")
            new_str = params.get("new_str", "")
            if not path:
                return "Error: path is required"
            if not old_str:
                return "Error: old_str is required"
            return await self._edit_file(path, old_str, new_str)

        return FunctionTool(
            name="edit_file",
            description=(
                "Edit a file by replacing an exact string match. "
                "The old_str must appear exactly once in the file."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to edit."},
                    "old_str": {"type": "string", "description": "The exact string to find and replace."},
                    "new_str": {"type": "string", "description": "The replacement string."},
                },
                "required": ["path", "old_str", "new_str"],
            },
            on_invoke_tool=edit_file,
        )

    # ------------------------------------------------------------------
    # Internal runtime methods
    # ------------------------------------------------------------------

    async def _run_bash(self, command: str, timeout: float | None = None) -> str:
        """Run a command in the persistent bash session."""
        from swerex.runtime.abstract import BashAction

        if self._runtime is None:
            return "Error: runtime is not initialized"

        try:
            obs = await self._runtime.run_in_session(
                BashAction(
                    command=command,
                    session=self._session_name,
                    timeout=timeout or self._bash_timeout,
                    check="silent",
                )
            )
        except Exception as e:
            logger.error("Bash command failed: %s", e)
            return f"Error: {e}"

        output = obs.output or ""
        if obs.exit_code and obs.exit_code != 0:
            output = f"{output}\n[exit code: {obs.exit_code}]"
        return _truncate(output)

    async def _read_file(self, path: str) -> str:
        """Read a file from the remote environment."""
        from swerex.runtime.abstract import ReadFileRequest

        if self._runtime is None:
            return "Error: runtime is not initialized"

        try:
            resp = await self._runtime.read_file(ReadFileRequest(path=path))
            return _truncate(resp.content)
        except Exception as e:
            logger.error("read_file failed for %s: %s", path, e)
            return f"Error reading {path}: {e}"

    async def _write_file(self, path: str, content: str) -> str:
        """Write a file in the remote environment."""
        from swerex.runtime.abstract import WriteFileRequest

        if self._runtime is None:
            return "Error: runtime is not initialized"

        try:
            await self._runtime.write_file(WriteFileRequest(path=path, content=content))
            return f"Successfully wrote to {path}"
        except Exception as e:
            logger.error("write_file failed for %s: %s", path, e)
            return f"Error writing {path}: {e}"

    async def _edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Edit a file by replacing an exact string occurrence."""
        if self._runtime is None:
            return "Error: runtime is not initialized"

        # Read → replace → write back.
        from swerex.runtime.abstract import ReadFileRequest, WriteFileRequest

        try:
            resp = await self._runtime.read_file(ReadFileRequest(path=path))
            content = resp.content
        except Exception as e:
            return f"Error reading {path}: {e}"

        count = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1:
            return f"Error: old_str found {count} times in {path} (expected exactly 1)"

        new_content = content.replace(old_str, new_str, 1)
        try:
            await self._runtime.write_file(WriteFileRequest(path=path, content=new_content))
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_json(input_json: str) -> dict:
    try:
        return json.loads(input_json) if input_json else {}
    except json.JSONDecodeError:
        return {}


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT_CHARS:
        return text[:_MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text
