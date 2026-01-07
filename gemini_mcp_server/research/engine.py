"""Research execution engine using Gemini with grounding."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gemini_mcp_server.client import RateLimitError, _is_rate_limit_error
from gemini_mcp_server.models import ResearchDepth, ResearchResult

if TYPE_CHECKING:
    from google import genai


# Depth-based configuration
DEPTH_CONFIG = {
    ResearchDepth.QUICK: {
        "model": "gemini-2.0-flash",
        "max_tokens": 1024,
        "prompt_prefix": "Provide a brief overview of:",
    },
    ResearchDepth.STANDARD: {
        "model": "gemini-2.0-flash",
        "max_tokens": 4096,
        "prompt_prefix": "Research and provide a comprehensive summary of:",
    },
    ResearchDepth.THOROUGH: {
        "model": "gemini-2.0-flash",
        "max_tokens": 8192,
        "prompt_prefix": "Conduct thorough research and provide a detailed analysis of:",
    },
}


def _build_research_prompt(topic: str, depth: ResearchDepth) -> str:
    """Build a research prompt based on depth."""
    config = DEPTH_CONFIG[depth]
    prefix = config["prompt_prefix"]

    return f"""{prefix} {topic}

Please include:
1. Key findings and main points
2. Relevant context and background
3. Current status or latest developments
4. Important considerations or caveats

Format the response clearly with sections if appropriate."""


def _extract_sources_from_response(response: str) -> list[str]:
    """Extract URLs from the response text."""
    # Simple URL extraction - could be enhanced
    url_pattern = r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:]'
    urls = re.findall(url_pattern, response)
    # Dedupe while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError,)),
    reraise=True,
)
async def execute_research(
    client: genai.Client,
    job_id: str,
    topic: str,
    depth: ResearchDepth,
) -> ResearchResult:
    """Execute deep research using Gemini with grounding.

    Args:
        client: Gemini API client
        job_id: Job identifier for the result
        topic: Research topic
        depth: Research depth level

    Returns:
        ResearchResult with summary and sources
    """
    config = DEPTH_CONFIG[depth]
    model = str(config["model"])
    prompt = _build_research_prompt(topic, depth)

    try:
        # Use Gemini with Google Search grounding
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=config["max_tokens"],
            ),
        )

        # Extract text from response
        if not response.candidates:
            return ResearchResult(
                job_id=job_id,
                summary="No response generated",
                sources=[],
                raw_response=None,
            )

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return ResearchResult(
                job_id=job_id,
                summary="No content in response",
                sources=[],
                raw_response=None,
            )

        # Collect all text parts
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        raw_text = "\n".join(text_parts)

        # Extract grounding metadata if available
        sources = []
        grounding_metadata = getattr(candidate, "grounding_metadata", None)
        if grounding_metadata:
            # Try to get sources from grounding chunks
            grounding_chunks = getattr(grounding_metadata, "grounding_chunks", [])
            for chunk in grounding_chunks or []:
                web_chunk = getattr(chunk, "web", None)
                if web_chunk:
                    uri = getattr(web_chunk, "uri", None)
                    if uri:
                        sources.append(uri)

            # Also check search entry point if available
            search_entry = getattr(grounding_metadata, "search_entry_point", None)
            if search_entry:
                rendered = getattr(search_entry, "rendered_content", None)
                if rendered:
                    # Extract URLs from rendered search widget
                    sources.extend(_extract_sources_from_response(rendered))

        # Fallback: extract URLs from response text
        if not sources:
            sources = _extract_sources_from_response(raw_text)

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)

        return ResearchResult(
            job_id=job_id,
            summary=raw_text,
            sources=unique_sources,
            raw_response=raw_text,
        )

    except Exception as e:
        if _is_rate_limit_error(e):
            raise RateLimitError(str(e)) from e
        raise
