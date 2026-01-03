"""Pydantic models for Gemini MCP Server."""

from pydantic import BaseModel, Field

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
