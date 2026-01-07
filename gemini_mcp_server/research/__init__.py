"""Research module for Gemini MCP Server.

This module provides async deep research capabilities that work for both:
- Option A: Claude Code via MCP tools (polling-based)
- Option B: pydantic-ai agent loops (direct Python API)

Example usage (Option B - Python API):
    ```python
    from gemini_mcp_server.research import ResearchManager, ResearchState

    manager = ResearchManager(db_path="research.db")
    job = await manager.start(topic="quantum computing")
    result = await manager.wait_for(job.id, timeout=300)
    print(result.result.summary)
    ```
"""

from gemini_mcp_server.models import (
    ResearchDepth,
    ResearchJob,
    ResearchResult,
    ResearchState,
    ResearchStatus,
)
from gemini_mcp_server.research.manager import ResearchManager
from gemini_mcp_server.research.store import ResearchStore
from gemini_mcp_server.research.worker import ResearchWorker

__all__ = [
    "ResearchDepth",
    "ResearchJob",
    "ResearchManager",
    "ResearchResult",
    "ResearchState",
    "ResearchStatus",
    "ResearchStore",
    "ResearchWorker",
]
