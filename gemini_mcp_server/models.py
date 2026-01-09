"""Pydantic models for Gemini MCP Server."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# --- Research Enums ---


class ResearchState(str, Enum):
    """State of a research job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchDepth(str, Enum):
    """How thorough the research should be."""

    QUICK = "quick"
    STANDARD = "standard"
    THOROUGH = "thorough"


# --- Research Models ---


class ResearchJob(BaseModel):
    """A background research job."""

    id: str = Field(description="Unique job identifier")
    topic: str = Field(description="Research topic")
    depth: ResearchDepth = Field(default=ResearchDepth.STANDARD)
    state: ResearchState = Field(default=ResearchState.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_complete(self) -> bool:
        return self.state == ResearchState.COMPLETED

    @property
    def is_terminal(self) -> bool:
        return self.state in (ResearchState.COMPLETED, ResearchState.FAILED)


class ResearchResult(BaseModel):
    """Result from completed research."""

    job_id: str
    summary: str = Field(description="Research summary")
    sources: list[str] = Field(default_factory=list)
    raw_response: str | None = None


class ResearchStatus(BaseModel):
    """Status check response."""

    job: ResearchJob
    result: ResearchResult | None = None
    error: str | None = None


# --- Research Tool Input Models ---


class StartResearchInput(BaseModel):
    """Input for start_research tool."""

    topic: str = Field(description="Research topic or question")
    depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD,
        description="How thorough the research should be",
    )


class GetResearchInput(BaseModel):
    """Input for get_research tool."""

    job_id: str = Field(description="Job ID from start_research")


class ListResearchInput(BaseModel):
    """Input for list_research tool."""

    state: ResearchState | None = Field(
        default=None,
        description="Filter by state (optional)",
    )
    limit: int = Field(default=10, ge=1, le=100)


# --- Tool Input Models ---


class AskGeminiInput(BaseModel):
    """Input schema for ask_gemini tool."""

    prompt: str = Field(description="The question or prompt for Gemini")
    model: str | None = Field(
        default=None,
        description="The Gemini model to use (default: gemini-2.0-flash). "
        "Use list_gemini_models to see available models.",
    )


class ListModelsInput(BaseModel):
    """Input schema for list_gemini_models tool."""

    pass


class RunCodeInput(BaseModel):
    """Input schema for run_code tool."""

    code: str = Field(description="Python code to execute")
    context: str | None = Field(
        default=None,
        description="Optional context about what the code should accomplish",
    )


class ReadFileInput(BaseModel):
    """Input schema for read_file tool."""

    file_path: str = Field(description="Absolute path to the file to read")
    prompt: str = Field(
        default="Analyze this file and summarize its contents.",
        description="What to do with the file contents (e.g., 'summarize', 'find bugs', 'explain')",
    )
    model: str | None = Field(
        default=None,
        description="The Gemini model to use (default: gemini-2.0-flash)",
    )


class ReadFilesInput(BaseModel):
    """Input schema for read_files tool."""

    file_paths: list[str] = Field(
        description="List of absolute paths to files to read",
        min_length=1,
        max_length=20,
    )
    prompt: str = Field(
        default="Analyze these files and summarize their contents.",
        description="What to do with the file contents (e.g., 'compare', 'find bugs', 'explain relationships')",
    )
    model: str | None = Field(
        default=None,
        description="The Gemini model to use (default: gemini-2.0-flash)",
    )


class WriteFileInput(BaseModel):
    """Input schema for write_file tool."""

    file_path: str = Field(
        description="Absolute path to the file to write. Must be within allowed writable directories."
    )
    content: str = Field(description="UTF-8 text content to write. Max 1MB.")
    create_directories: bool = Field(
        default=False,
        description="Create parent directories if they don't exist",
    )


class GenerateImageInput(BaseModel):
    """Input schema for generate_image tool."""

    prompt: str = Field(description="Description of the image to generate")
    model: str | None = Field(
        default=None,
        description="Model to use: gemini-2.5-flash-image (default, fast) "
        "or gemini-3-pro-image-preview (higher quality)",
    )
    aspect_ratio: str = Field(
        default="1:1",
        description="Aspect ratio: 1:1, 16:9, 9:16, 4:3, 3:4",
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory to save image (default: /tmp)",
    )


# --- Response Models ---


class TokenUsage(BaseModel):
    """Token usage tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GeminiResponse(BaseModel):
    """Standard response from Gemini."""

    content: str
    model: str | None = None
    usage: TokenUsage | None = None


class CodeExecutionResult(BaseModel):
    """Structured output from code execution."""

    success: bool
    output: str | None = None
    error: str | None = None
    code_executed: str | None = Field(
        default=None,
        description="The actual code that was executed (may differ from input)",
    )


class ImageGenerationResult(BaseModel):
    """Structured output from image generation."""

    success: bool
    file_path: str | None = None
    error: str | None = None
    model_used: str | None = None
