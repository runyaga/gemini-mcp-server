"""Retry-enabled client helpers for Gemini API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gemini_mcp_server.models import CodeExecutionResult, ImageGenerationResult

if TYPE_CHECKING:
    from google import genai


class RateLimitError(Exception):
    """Rate limit exceeded."""

    pass


class GeminiAPIError(Exception):
    """General Gemini API error."""

    pass


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Check if exception is a rate limit error."""
    # google-genai raises various exceptions for rate limits
    error_str = str(exc).lower()
    return any(
        indicator in error_str
        for indicator in ["429", "rate limit", "quota", "resource exhausted"]
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError,)),
    reraise=True,
)
async def generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    **kwargs: Any,
) -> Any:
    """Generate content with automatic retry on rate limits.

    Args:
        client: Gemini API client
        model: Model name (e.g., 'gemini-2.0-flash')
        contents: The prompt/content to send
        **kwargs: Additional arguments for generate_content

    Returns:
        GenerateContentResponse from Gemini

    Raises:
        RateLimitError: If rate limit exceeded after retries
        GeminiAPIError: For other API errors
    """
    try:
        return await client.aio.models.generate_content(
            model=model,
            contents=contents,
            **kwargs,
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            raise RateLimitError(str(e)) from e
        raise GeminiAPIError(str(e)) from e


def parse_code_execution_response(response: Any) -> CodeExecutionResult:
    """Parse code execution results from Gemini response.

    Gemini returns executable_code and code_execution_result parts
    when code_execution tool is enabled.

    Args:
        response: GenerateContentResponse from Gemini

    Returns:
        Structured CodeExecutionResult
    """
    code_executed = None
    output = None
    error = None
    success = False

    # Extract parts from response
    if not response.candidates:
        return CodeExecutionResult(
            success=False,
            error="No response candidates",
        )

    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return CodeExecutionResult(
            success=False,
            error="No content parts in response",
        )

    for part in candidate.content.parts:
        # Check for executed code
        if hasattr(part, "executable_code") and part.executable_code:
            code_executed = part.executable_code.code

        # Check for execution result
        if hasattr(part, "code_execution_result") and part.code_execution_result:
            result = part.code_execution_result
            outcome = getattr(result, "outcome", None)

            # Outcome can be string or enum
            outcome_str = str(outcome).upper() if outcome else ""
            success = "OK" in outcome_str or outcome_str == "OUTCOME_OK"

            output = getattr(result, "output", None)
            if not success and output:
                error = output
                output = None

    # If no code execution parts found, check for text response
    if code_executed is None:
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                return CodeExecutionResult(
                    success=False,
                    error=f"No code executed. Model response: {part.text}",
                )

    return CodeExecutionResult(
        success=success,
        output=output,
        error=error,
        code_executed=code_executed,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError,)),
    reraise=True,
)
async def execute_code(
    client: genai.Client,
    code: str,
    model: str = "gemini-2.0-flash",
    context: str | None = None,
) -> CodeExecutionResult:
    """Execute Python code using Gemini's code execution capability.

    Args:
        client: Gemini API client
        code: Python code to execute
        model: Model to use (default: gemini-2.0-flash)
        context: Optional context about what the code should do

    Returns:
        CodeExecutionResult with output or error
    """
    # Build prompt
    if context:
        prompt = f"{context}\n\nExecute this Python code:\n```python\n{code}\n```"
    else:
        prompt = (
            f"Execute this Python code and return the output:\n```python\n{code}\n```"
        )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(code_execution=types.ToolCodeExecution())],
            ),
        )
        return parse_code_execution_response(response)
    except Exception as e:
        if _is_rate_limit_error(e):
            raise RateLimitError(str(e)) from e
        return CodeExecutionResult(
            success=False,
            error=str(e),
        )


DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError,)),
    reraise=True,
)
async def generate_image(
    client: genai.Client,
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    aspect_ratio: str = "1:1",
    output_path: str | None = None,
) -> ImageGenerationResult:
    """Generate an image using Gemini's Nano Banana capability.

    Args:
        client: Gemini API client
        prompt: Description of the image to generate
        model: Model to use (gemini-2.5-flash-image or gemini-3-pro-image-preview)
        aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
        output_path: Path to save the image (auto-generated if None)

    Returns:
        ImageGenerationResult with file path or error
    """
    import uuid
    from pathlib import Path

    # Generate output path if not provided
    if output_path is None:
        output_path = f"/tmp/gemini_img_{uuid.uuid4().hex[:8]}.png"

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                aspect_ratio=aspect_ratio,
            ),
        )

        # Extract image from response
        if not response.candidates:
            return ImageGenerationResult(
                success=False,
                error="No response candidates",
                model_used=model,
            )

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return ImageGenerationResult(
                success=False,
                error="No content parts in response",
                model_used=model,
            )

        # Find the image part
        for part in candidate.content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                # Save the image using PIL via the as_image() method if available
                # or decode base64 manually
                try:
                    image = part.as_image()
                    image.save(output_path)
                except AttributeError:
                    # Fallback: decode base64 manually
                    import base64

                    image_data = base64.b64decode(part.inline_data.data)
                    Path(output_path).write_bytes(image_data)

                return ImageGenerationResult(
                    success=True,
                    file_path=output_path,
                    model_used=model,
                )

        # No image found, check for text error
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                return ImageGenerationResult(
                    success=False,
                    error=f"No image generated. Model response: {part.text}",
                    model_used=model,
                )

        return ImageGenerationResult(
            success=False,
            error="No image data in response",
            model_used=model,
        )

    except Exception as e:
        if _is_rate_limit_error(e):
            raise RateLimitError(str(e)) from e
        return ImageGenerationResult(
            success=False,
            error=str(e),
            model_used=model,
        )
