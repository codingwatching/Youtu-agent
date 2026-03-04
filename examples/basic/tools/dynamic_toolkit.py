"""Dynamic toolkit that registers tools from an API at runtime.

Example usage:
    toolkit = DynamicToolkit(config=ToolkitConfig(config={"base_url": "http://localhost:8000/v1"}))
    await toolkit.build()
    # Now tools are available via toolkit.tools_map
"""

import json
from typing import Any

import httpx

from utu.config import ToolkitConfig
from utu.utils import get_logger
from utu.tools.base import AsyncBaseToolkit

logger = get_logger(__name__)


class DynamicToolkit(AsyncBaseToolkit):
    """Toolkit that dynamically registers tools from an API endpoint.

    On initialization, calls ${base_url}/init to get session_id and tool_schemas.
    Tools are then dynamically created and registered based on the schemas.
    Tool invocations are forwarded to ${base_url}/call.
    """

    def __init__(self, config: ToolkitConfig | dict | None = None) -> None:
        super().__init__(config)
        self.base_url = self.config.config.get("base_url", "http://localhost:8000/v1")
        self.timeout = self.config.config.get("timeout", 60)
        self.session_id: str | None = None
        self._initialized = False

    async def build(self) -> None:
        """Initialize session by calling the /init endpoint and registering tools."""
        if self._initialized:
            logger.warning("Session already initialized, skipping.")
            return

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/init",
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        self.session_id = data.get("session_id")
        tool_schemas = data.get("tool_schemas", [])

        logger.info(f"Initialized session {self.session_id} with {len(tool_schemas)} tools")

        for schema in tool_schemas:
            self._register_dynamic_tool(schema)

        self._initialized = True

    def _register_dynamic_tool(self, schema: dict) -> None:
        """Dynamically create and register a tool from its schema.

        Args:
            schema: Tool schema containing name, description, and inputSchema
        """
        tool_name = schema["name"]
        description = schema.get("description", "")
        input_schema = schema.get("inputSchema", {})

        # Build docstring with argument descriptions for the LLM
        docstring = self._build_docstring(description, input_schema)

        # Create the async tool function using a closure to capture tool_name
        def create_tool_func(name: str, doc: str):
            async def tool_func(**kwargs) -> str:
                return await self._call_remote_tool(name, kwargs)

            tool_func.__name__ = name
            tool_func.__doc__ = doc
            tool_func._is_tool = True
            tool_func._tool_name = name
            return tool_func

        tool_func = create_tool_func(tool_name, docstring)

        # Set as instance attribute so tools_map can find it
        setattr(self, tool_name, tool_func)

        # Also add to _tools_map directly if it exists
        if self._tools_map is not None:
            self._tools_map[tool_name] = tool_func

        logger.debug(f"Registered dynamic tool: {tool_name}")

    def _build_docstring(self, description: str, input_schema: dict) -> str:
        """Build a docstring from the tool description and input schema.

        Args:
            description: Tool description
            input_schema: JSON schema for the tool's input parameters

        Returns:
            Formatted docstring with Args section
        """
        lines = [description, "", "Args:"]

        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        for param_name, param_info in properties.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "any")
            is_required = param_name in required
            default = param_info.get("default")

            # Format: param_name (type, required/optional): description
            req_str = "required" if is_required else "optional"
            if default is not None:
                req_str += f", default={default}"

            lines.append(f"    {param_name} ({param_type}, {req_str}): {param_desc}")

        return "\n".join(lines)

    async def _call_remote_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool via the remote API.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/call",
                json={
                    "session_id": self.session_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

        # Handle different response formats
        if isinstance(result, dict):
            if "result" in result:
                return self._format_result(result["result"])
            elif "error" in result:
                return f"Error: {result['error']}"
            else:
                return json.dumps(result, ensure_ascii=False)
        return str(result)

    def _format_result(self, result: Any) -> str:
        """Format the tool result for output.

        Args:
            result: Raw result from the API

        Returns:
            Formatted string result
        """
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    async def cleanup(self) -> None:
        """Cleanup resources. Can be extended to call a /cleanup endpoint if needed."""
        self._initialized = False
        self.session_id = None
        self._tools_map = None
