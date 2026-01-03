"""Streaming support for Gemini MCP Server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from gemini_mcp_server.deps import GeminiDeps


async def stream_agent_response(
    agent: Agent[GeminiDeps, str],
    prompt: str,
    deps: GeminiDeps,
    progress_token: str | int | None = None,
    session: Any | None = None,
) -> str:
    """Stream response from agent with optional MCP progress notifications.

    Args:
        agent: The pydantic-ai agent to run
        prompt: The prompt to send
        deps: Dependencies for the agent
        progress_token: MCP progress token (if client supports progress)
        session: MCP session for sending progress notifications

    Returns:
        Complete response text
    """
    chunks: list[str] = []

    async with agent.run_stream(prompt, deps=deps) as result:
        async for text in result.stream_text(delta=True):
            chunks.append(text)

            # Send progress notification if client supports it
            # Note: contextlib.suppress doesn't work with async await
            if progress_token is not None and session is not None:
                try:  # noqa: SIM105
                    await session.send_progress_notification(
                        progress_token=progress_token,
                        progress=len(chunks),
                        message=text,
                    )
                except Exception:
                    # Progress notifications are optional - don't fail the request
                    pass

    return "".join(chunks)


async def run_with_optional_streaming(
    agent: Agent[GeminiDeps, str],
    prompt: str,
    deps: GeminiDeps,
    progress_token: str | int | None = None,
    session: Any | None = None,
) -> str:
    """Run agent with streaming if progress token provided, otherwise regular run.

    This is the main entry point for tool implementations.

    Args:
        agent: The pydantic-ai agent to run
        prompt: The prompt to send
        deps: Dependencies for the agent
        progress_token: MCP progress token (if client supports progress)
        session: MCP session for sending progress notifications

    Returns:
        Complete response text
    """
    if progress_token is not None and session is not None:
        return await stream_agent_response(
            agent=agent,
            prompt=prompt,
            deps=deps,
            progress_token=progress_token,
            session=session,
        )

    # Non-streaming fallback
    result = await agent.run(prompt, deps=deps)
    return result.output
