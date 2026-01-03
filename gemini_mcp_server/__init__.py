"""Gemini MCP Server using pydantic-ai."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from gemini_mcp_server.client import execute_code
from gemini_mcp_server.deps import GeminiDeps
from gemini_mcp_server.models import AskGeminiInput, ListModelsInput, RunCodeInput

load_dotenv()

# Initialize MCP server
server = Server("gemini-mcp")

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
DEFAULT_MODEL = "gemini-2.0-flash"


# --- Agent Factory ---


@lru_cache(maxsize=10)
def get_provider() -> GoogleProvider | None:
    """Get cached Google provider."""
    return GoogleProvider(api_key=API_KEY) if API_KEY else None


def create_agent(model_name: str | None = None) -> Agent[GeminiDeps, str]:
    """Create an agent with the specified model.

    Args:
        model_name: Model to use (defaults to gemini-2.0-flash)

    Returns:
        Configured pydantic-ai Agent
    """
    provider = get_provider()
    if provider is None:
        model = GoogleModel(model_name or DEFAULT_MODEL)
    else:
        model = GoogleModel(model_name or DEFAULT_MODEL, provider=provider)

    return Agent(
        model,
        deps_type=GeminiDeps,
        output_type=str,
        system_prompt="You are a helpful assistant. Provide clear, accurate, and concise responses.",
    )


# --- MCP Tool Handlers ---


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="ask_gemini",
            description="Ask Gemini a question. Use for research, current info, or when you need Google's perspective.",
            inputSchema=AskGeminiInput.model_json_schema(),
        ),
        Tool(
            name="list_gemini_models",
            description="List available Gemini models.",
            inputSchema=ListModelsInput.model_json_schema(),
        ),
        Tool(
            name="run_code",
            description="Execute Python code using Gemini's sandboxed code execution. "
            "Supports numpy, pandas, and standard library. No pip install.",
            inputSchema=RunCodeInput.model_json_schema(),
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "ask_gemini":
        inputs = AskGeminiInput.model_validate(arguments)
        agent = create_agent(inputs.model)
        deps = GeminiDeps.from_env()

        # Run agent (streaming infrastructure is ready for future MCP progress support)
        result = await agent.run(inputs.prompt, deps=deps)
        response = result.output
        return [TextContent(type="text", text=response)]

    if name == "list_gemini_models":
        ListModelsInput.model_validate(arguments)
        deps = GeminiDeps.from_env()
        models = [
            f"{m.name}: {m.display_name}"
            for m in deps.client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        return [TextContent(type="text", text="\n".join(models))]

    if name == "run_code":
        code_inputs = RunCodeInput.model_validate(arguments)
        code_deps = GeminiDeps.from_env()
        exec_result = await execute_code(
            client=code_deps.client,
            code=code_inputs.code,
            model=code_deps.default_model,
            context=code_inputs.context,
        )

        # Format output
        if exec_result.success:
            output = exec_result.output or "(no output)"
            response = f"✓ Code executed successfully\n\nOutput:\n{output}"
        else:
            response = f"✗ Execution failed\n\nError:\n{exec_result.error}"

        if exec_result.code_executed and exec_result.code_executed != code_inputs.code:
            response += (
                f"\n\nCode executed:\n```python\n{exec_result.code_executed}\n```"
            )

        return [TextContent(type="text", text=response)]

    raise ValueError(f"Unknown tool: {name}")


async def _run() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    """Entry point for the MCP server."""
    import asyncio

    asyncio.run(_run())


# Re-export for backwards compatibility with tests
__all__ = [
    "AskGeminiInput",
    "GeminiDeps",
    "ListModelsInput",
    "RunCodeInput",
    "call_tool",
    "create_agent",
    "execute_code",
    "list_tools",
    "main",
    "server",
]


if __name__ == "__main__":
    main()
