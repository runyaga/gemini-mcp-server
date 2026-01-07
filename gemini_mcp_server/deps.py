"""Dependencies for Gemini MCP Server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai

if TYPE_CHECKING:
    pass


@dataclass
class GeminiDeps:
    """Dependencies injected into Gemini agents and tools.

    The pydantic-ai Agent manages its own LLM client internally based on the model.
    This client is only needed for direct API calls:
    - list_models (client.models.list)
    - run_code (generate_content with code_execution tool)
    - deep_research (interactions API)
    """

    client: genai.Client
    default_model: str = "gemini-2.0-flash"
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> GeminiDeps:
        """Create deps from environment variables."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        return cls(client=genai.Client(api_key=api_key))
