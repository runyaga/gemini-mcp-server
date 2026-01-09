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

from gemini_mcp_server.client import execute_code, generate_image
from gemini_mcp_server.config import get_config
from gemini_mcp_server.deps import GeminiDeps
from gemini_mcp_server.models import (
    AskGeminiInput,
    GenerateImageInput,
    GetResearchInput,
    ListModelsInput,
    ListResearchInput,
    ReadFileInput,
    ReadFilesInput,
    RunCodeInput,
    StartResearchInput,
    WriteFileInput,
)
from gemini_mcp_server.research import ResearchManager

load_dotenv()

# Initialize MCP server
server = Server("gemini-mcp")

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
DEFAULT_MODEL = "gemini-2.0-flash"
RESEARCH_DB_PATH = os.environ.get("GEMINI_RESEARCH_DB", "research.db")

# read_files limits
MAX_FILES = 20
MAX_TOTAL_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

# Singleton research manager (lazy initialized)
_research_manager: ResearchManager | None = None


def get_research_manager() -> ResearchManager:
    """Get or create the research manager singleton."""
    global _research_manager
    if _research_manager is None:
        _research_manager = ResearchManager(db_path=RESEARCH_DB_PATH)
    return _research_manager


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
        Tool(
            name="start_research",
            description="Start a background deep research job using Gemini with Google Search grounding. "
            "Returns immediately with a job_id. Use get_research to check status and retrieve results.",
            inputSchema=StartResearchInput.model_json_schema(),
        ),
        Tool(
            name="get_research",
            description="Get the status and result of a research job. "
            "Returns job state, and if completed, the research summary and sources.",
            inputSchema=GetResearchInput.model_json_schema(),
        ),
        Tool(
            name="list_research",
            description="List research jobs, optionally filtered by state. "
            "Returns jobs ordered by creation time (newest first).",
            inputSchema=ListResearchInput.model_json_schema(),
        ),
        Tool(
            name="generate_image",
            description="Generate an image using Gemini's Nano Banana capability. "
            "Returns the file path to the generated image.",
            inputSchema=GenerateImageInput.model_json_schema(),
        ),
        Tool(
            name="read_file",
            description="Read a local file and have Gemini analyze it. "
            "Use this to offload file analysis to Gemini, saving context in your conversation. "
            "Supports code, text, data files, etc.",
            inputSchema=ReadFileInput.model_json_schema(),
        ),
        Tool(
            name="read_files",
            description="Read multiple local files and have Gemini analyze them together. "
            "Use this to offload batch file analysis to Gemini. "
            "Max 20 files, 5MB total size.",
            inputSchema=ReadFilesInput.model_json_schema(),
        ),
        Tool(
            name="write_file",
            description="Write content to a local file. "
            "Use this to save results from analysis or generation. "
            "Requires writable directories to be configured.",
            inputSchema=WriteFileInput.model_json_schema(),
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

    if name == "start_research":
        research_input = StartResearchInput.model_validate(arguments)
        manager = get_research_manager()
        job = await manager.start(
            topic=research_input.topic, depth=research_input.depth
        )
        response = (
            f"Research job started.\n\n"
            f"Job ID: {job.id}\n"
            f"Topic: {job.topic}\n"
            f"Depth: {job.depth.value}\n"
            f"State: {job.state.value}\n\n"
            f"Use get_research with job_id='{job.id}' to check status and retrieve results."
        )
        return [TextContent(type="text", text=response)]

    if name == "get_research":
        get_input = GetResearchInput.model_validate(arguments)
        manager = get_research_manager()
        status = await manager.get_status(get_input.job_id)

        if status is None:
            return [TextContent(type="text", text=f"Job not found: {get_input.job_id}")]

        response_lines = [
            f"Job ID: {status.job.id}",
            f"Topic: {status.job.topic}",
            f"Depth: {status.job.depth.value}",
            f"State: {status.job.state.value}",
            f"Created: {status.job.created_at.isoformat()}",
        ]

        if status.job.started_at:
            response_lines.append(f"Started: {status.job.started_at.isoformat()}")
        if status.job.completed_at:
            response_lines.append(f"Completed: {status.job.completed_at.isoformat()}")

        if status.result:
            response_lines.append("")
            response_lines.append("--- Research Summary ---")
            response_lines.append(status.result.summary)
            if status.result.sources:
                response_lines.append("")
                response_lines.append("--- Sources ---")
                for source in status.result.sources:
                    response_lines.append(f"- {source}")

        if status.error:
            response_lines.append("")
            response_lines.append(f"Error: {status.error}")

        return [TextContent(type="text", text="\n".join(response_lines))]

    if name == "list_research":
        list_input = ListResearchInput.model_validate(arguments)
        manager = get_research_manager()
        jobs = await manager.list_jobs(state=list_input.state, limit=list_input.limit)

        if not jobs:
            filter_msg = (
                f" with state '{list_input.state.value}'" if list_input.state else ""
            )
            return [
                TextContent(type="text", text=f"No research jobs found{filter_msg}.")
            ]

        response_lines = [f"Found {len(jobs)} job(s):", ""]
        for job in jobs:
            response_lines.append(
                f"- {job.id[:8]}... | {job.state.value:10} | {job.topic[:50]}"
            )

        return [TextContent(type="text", text="\n".join(response_lines))]

    if name == "generate_image":
        image_input = GenerateImageInput.model_validate(arguments)
        image_deps = GeminiDeps.from_env()

        # Determine output path
        output_path = None
        if image_input.output_dir:
            import uuid

            output_path = (
                f"{image_input.output_dir}/gemini_img_{uuid.uuid4().hex[:8]}.png"
            )

        image_result = await generate_image(
            client=image_deps.client,
            prompt=image_input.prompt,
            model=image_input.model or "gemini-2.5-flash-image",
            aspect_ratio=image_input.aspect_ratio,
            output_path=output_path,
        )

        if image_result.success:
            response = (
                f"Image generated successfully.\n\n"
                f"File: {image_result.file_path}\n"
                f"Model: {image_result.model_used}"
            )
        else:
            response = f"Image generation failed.\n\nError: {image_result.error}"

        return [TextContent(type="text", text=response)]

    if name == "read_file":
        from pathlib import Path

        file_input = ReadFileInput.model_validate(arguments)
        file_path = Path(file_input.file_path)

        # Security: Validate path against allowed directories and deny patterns
        config = get_config()
        is_allowed, error_msg = config.read_file.is_path_allowed(file_path)
        if not is_allowed:
            return [TextContent(type="text", text=f"Error: {error_msg}")]

        # Resolve to canonical path after validation
        resolved_path = file_path.expanduser().resolve()

        # Validate is a file (not directory)
        if not resolved_path.is_file():
            return [TextContent(type="text", text=f"Error: Not a file: {file_path}")]

        # Read file contents
        try:
            content = resolved_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Cannot read binary file as text: {file_path}",
                )
            ]
        except PermissionError:
            return [
                TextContent(type="text", text=f"Error: Permission denied: {file_path}")
            ]

        # Build prompt with file contents
        file_prompt = (
            f"{file_input.prompt}\n\n"
            f"--- File: {resolved_path.name} ---\n"
            f"{content}\n"
            f"--- End of file ---"
        )

        # Send to Gemini
        agent = create_agent(file_input.model)
        deps = GeminiDeps.from_env()
        result = await agent.run(file_prompt, deps=deps)

        response = f"File: {resolved_path}\n\n{result.output}"
        return [TextContent(type="text", text=response)]

    if name == "read_files":
        from pathlib import Path

        files_input = ReadFilesInput.model_validate(arguments)
        config = get_config()

        # Collect file contents and track errors
        file_contents: list[tuple[Path, str]] = []
        errors: list[str] = []
        total_size = 0

        for file_path_str in files_input.file_paths:
            file_path = Path(file_path_str)

            # Security: Validate path against allowed directories and deny patterns
            is_allowed, error_msg = config.read_file.is_path_allowed(file_path)
            if not is_allowed:
                errors.append(f"{file_path}: {error_msg}")
                continue

            # Resolve to canonical path after validation
            resolved_path = file_path.expanduser().resolve()

            # Validate is a file (not directory)
            if not resolved_path.is_file():
                errors.append(f"{file_path}: Not a file")
                continue

            # Read file contents
            try:
                content = resolved_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"{file_path}: Cannot read binary file as text")
                continue
            except PermissionError:
                errors.append(f"{file_path}: Permission denied")
                continue

            # Track total size
            content_size = len(content.encode("utf-8"))
            if total_size + content_size > MAX_TOTAL_SIZE_BYTES:
                errors.append(
                    f"{file_path}: Would exceed {MAX_TOTAL_SIZE_BYTES // (1024 * 1024)}MB total size limit"
                )
                continue

            total_size += content_size
            file_contents.append((resolved_path, content))

        # If all files failed, return error
        if not file_contents:
            error_list = "\n".join(f"  - {e}" for e in errors)
            return [
                TextContent(
                    type="text",
                    text=f"Error: Could not read any files:\n{error_list}",
                )
            ]

        # Build combined prompt with labeled files
        parts = [files_input.prompt, ""]
        for resolved_path, content in file_contents:
            parts.append(f'<file name="{resolved_path}">')
            parts.append(content)
            parts.append("</file>")
            parts.append("")

        file_prompt = "\n".join(parts)

        # Send to Gemini
        agent = create_agent(files_input.model)
        deps = GeminiDeps.from_env()
        result = await agent.run(file_prompt, deps=deps)

        # Build response
        files_read = [str(p) for p, _ in file_contents]
        response_parts = [
            f"Files analyzed ({len(file_contents)}):",
            *[f"  - {f}" for f in files_read],
        ]
        if errors:
            response_parts.append("")
            response_parts.append(f"Errors ({len(errors)}):")
            response_parts.extend(f"  - {e}" for e in errors)
        response_parts.append("")
        response_parts.append("--- Analysis ---")
        response_parts.append(result.output)

        return [TextContent(type="text", text="\n".join(response_parts))]

    if name == "write_file":
        from pathlib import Path

        write_input = WriteFileInput.model_validate(arguments)
        file_path = Path(write_input.file_path)
        config = get_config()

        # Check content size limit
        content_bytes = write_input.content.encode("utf-8")
        if len(content_bytes) > config.write_file.max_size_bytes:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Content size ({len(content_bytes)} bytes) exceeds limit ({config.write_file.max_size_bytes} bytes)",
                )
            ]

        # Security: Validate path against writable directories and deny patterns
        is_allowed, error_msg = config.write_file.is_write_allowed(file_path)
        if not is_allowed:
            return [TextContent(type="text", text=f"Error: {error_msg}")]

        # Resolve path (strict=False since file may not exist)
        resolved_path = file_path.expanduser().resolve()

        # Create parent directories if requested
        if write_input.create_directories:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved_path.parent.exists():
            return [
                TextContent(
                    type="text",
                    text=f"Error: Directory does not exist: {resolved_path.parent}",
                )
            ]

        # Write file
        try:
            resolved_path.write_bytes(content_bytes)
        except PermissionError:
            return [
                TextContent(type="text", text=f"Error: Permission denied: {file_path}")
            ]
        except OSError as e:
            return [TextContent(type="text", text=f"Error: {e}")]

        return [
            TextContent(
                type="text",
                text=f"Successfully wrote {len(content_bytes)} bytes to {resolved_path}",
            )
        ]

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
    "GenerateImageInput",
    "GetResearchInput",
    "ListModelsInput",
    "ListResearchInput",
    "ReadFileInput",
    "ReadFilesInput",
    "ResearchManager",
    "RunCodeInput",
    "StartResearchInput",
    "WriteFileInput",
    "call_tool",
    "create_agent",
    "execute_code",
    "generate_image",
    "get_config",
    "get_research_manager",
    "list_tools",
    "main",
    "server",
]


if __name__ == "__main__":
    main()
