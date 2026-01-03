# Implementation Plan: Deep Research & Code Execution Tools

## Overview

Add two new agentic tools to the Gemini MCP server:
1. **`deep_research`** - Long-running autonomous research via Gemini's Interactions API
2. **`run_code`** - Execute Python code via Gemini's code execution capability

---

## Idiomatic Pydantic-AI Patterns

### Key Principles

1. **Dependency Injection** (`deps_type`) - Pass config, clients, DB connections via dataclass
2. **Structured Output** (`output_type`) - Use Pydantic models for typed, validated responses
3. **Tools with Context** (`@agent.tool`) - Access deps via `RunContext[Deps]`
4. **Dynamic Instructions** (`@agent.instructions`) - Generate prompts from deps
5. **Testing** (`TestModel`) - Mock LLM responses without real API calls

### Our Dependencies Pattern

**Important:** The pydantic-ai `Agent` manages its own LLM client internally based on the model string (e.g., `'google-gla:gemini-2.0-flash'`). The `client` in `GeminiDeps` is **only** needed when tools make direct API calls (code execution, file uploads, Interactions API).

```python
from dataclasses import dataclass
from google import genai
from pydantic_ai import Agent, RunContext

@dataclass
class GeminiDeps:
    """Dependencies injected into Gemini tools that need direct API access."""
    client: genai.Client  # For code_execution, deep_research (direct API calls)
    default_model: str = "gemini-2.0-flash"
    job_manager: "ResearchJobManager | None" = None

# Type alias for our agents
GeminiAgent = Agent[GeminiDeps, str]
```

**When to use `deps.client`:**
| Tool | Needs `deps.client`? | Reason |
|------|---------------------|--------|
| `ask_gemini` | No | Agent handles inference internally |
| `list_models` | Yes | Direct `client.models.list()` call |
| `run_code` | Yes | Direct `generate_content` with `code_execution` tool |
| `deep_research` | Yes | Direct `client.interactions.create()` call |

### Structured Output Pattern

```python
from pydantic import BaseModel, Field

class CodeExecutionResult(BaseModel):
    """Structured output from code execution."""
    success: bool
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None

# Agent with typed output
code_agent: Agent[GeminiDeps, CodeExecutionResult] = Agent(
    'google-gla:gemini-2.0-flash',
    deps_type=GeminiDeps,
    output_type=CodeExecutionResult,
    system_prompt="Execute Python code and return structured results.",
)
```

### Tool Pattern with RunContext

```python
@code_agent.tool
async def execute_code(ctx: RunContext[GeminiDeps], code: str) -> str:
    """Execute Python code in Gemini's sandbox."""
    # Access deps through ctx.deps
    response = await ctx.deps.client.models.generate_content(
        model=ctx.deps.default_model,
        contents=code,
        config={"tools": [{"code_execution": {}}]}
    )
    return parse_code_result(response)
```

### Testing Pattern with TestModel

```python
import pytest
from pydantic_ai.models.test import TestModel

@pytest.fixture
def mock_agent():
    """Provide agent with mocked LLM responses."""
    return TestModel(custom_output_text='{"success": true, "stdout": "42"}')

async def test_code_execution(mock_agent):
    async with code_agent.override(model=mock_agent):
        deps = GeminiDeps(client=Mock(), default_model="test")
        result = await code_agent.run("print(42)", deps=deps)
        assert result.output.success is True
```

### Streaming Pattern (pydantic-ai)

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-2.0-flash')

async def stream_response(prompt: str):
    """Stream text output from agent."""
    async with agent.run_stream(prompt) as result:
        async for text in result.stream_text(delta=True):
            yield text  # Yield incremental deltas
```

### MCP Progress Notifications

MCP supports progress tracking for long-running operations:

```python
from mcp.server import Server
from mcp.types import TextContent

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Get progress token from request context (if provided)
    ctx = server.request_context
    progress_token = ctx.meta.get("progressToken") if ctx.meta else None

    if name == "ask_gemini" and progress_token:
        # Stream with progress notifications
        chunks = []
        async with agent.run_stream(prompt, deps=deps) as result:
            async for text in result.stream_text(delta=True):
                chunks.append(text)
                # Send progress notification
                await ctx.session.send_progress_notification(
                    progress_token=progress_token,
                    progress=len(chunks),
                    message=text
                )
        return [TextContent(type="text", text="".join(chunks))]
```

**Note:** Progress notifications require client support (progressToken in request).

---

## Best Practices: Tooling & Quality

### Ruff Configuration

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
required-version = ">=0.8.0"  # Pin for reproducibility

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "ERA",    # eradicate (commented code)
    "RUF",    # Ruff-specific rules
]
ignore = [
    "E501",   # line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__
"tests/*" = ["ARG"]       # Allow unused args in tests

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### pytest Configuration

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
markers = [
    "integration: marks tests as integration tests (require API key)",
    "slow: marks tests as slow running",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["gemini_mcp_server"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
fail_under = 80
```

### Type Checking (pyright/mypy)

```toml
[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "basic"
reportMissingTypeStubs = false

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

---

## Architecture Considerations

### Challenge: MCP Tool Timeout
- MCP tools are typically synchronous with short timeouts
- Deep Research takes 2-10+ minutes
- **Solution**: Return a job ID immediately, provide `get_research_status` and `cancel_research` tools

### Challenge: Server Restarts
- MCP servers restart frequently (IDE restarts, sleep, etc.)
- In-memory state is lost
- **Solution**: SQLite persistence for job state

### Challenge: Rate Limits
- Gemini has strict RPM limits on free/lower tiers
- Polling can quickly exhaust quota
- **Solution**: Built-in retry with exponential backoff from day 1

### API Requirements
- Deep Research: Requires Interactions API (`client.interactions.create`)
- Code Execution: Uses standard `generate_content` with `tools=[{"code_execution": {}}]`

---

## Milestone 0: Tooling & Project Setup

**Goal**: Set up proper development tooling before writing code

### Tasks
1. Update `pyproject.toml` with ruff, pytest, coverage config
2. Add dev dependencies (ruff, pytest-asyncio, pytest-cov, pre-commit)
3. Create `.pre-commit-config.yaml`
4. Run ruff on existing code and fix issues
5. Ensure existing tests still pass

### Updated pyproject.toml

```toml
[project]
name = "gemini-mcp-server"
version = "0.2.0"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.0",
    "pydantic>=2.0",
    "pydantic-ai>=0.1.0,<1.0",  # Pin - evolving rapidly
    "python-dotenv>=1.0",
    "google-genai>=1.0",
    "tenacity>=8.0",
    "aiosqlite>=0.19",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=4.0",
    "ruff>=0.8",
    "mypy>=1.0",              # Static type checking
    "pre-commit>=3.0",
]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "C4", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/*" = ["ARG"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
markers = [
    "integration: integration tests (require API key)",
]

[tool.coverage.run]
source = ["gemini_mcp_server"]
branch = true

[tool.coverage.report]
fail_under = 80

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "google.*"
ignore_missing_imports = true
```

### Deliverables
- [ ] Updated `pyproject.toml` with all tooling config
- [ ] `.pre-commit-config.yaml` created
- [ ] `uv pip install -e ".[dev]"` works
- [ ] `ruff check .` passes (or issues fixed)
- [ ] `ruff format .` applied
- [ ] `mypy gemini_mcp_server` passes (or issues fixed)
- [ ] `pytest -v` passes
- [ ] `pre-commit install` configured

---

## Milestone 1: MCP Skeleton & Connectivity

**Goal**: Verify MCP server works before adding AI complexity

### Why This Milestone?
Building AI logic before verifying MCP connectivity risks integration issues. Start with a simple "echo" tool to confirm:
- stdio transport works
- Claude Code / Claude Desktop can connect
- Tool schema is valid
- Response format is correct

### Tasks
1. Create minimal MCP server with `echo` tool
2. Test connectivity with Claude Code
3. Verify tool appears in `/mcp list`
4. Confirm tool can be called and returns response

### Implementation

```python
# gemini_mcp_server/__init__.py (minimal skeleton)
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

server = Server("gemini-mcp")

class EchoInput(BaseModel):
    """Input for echo tool."""
    message: str = Field(description="Message to echo back")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="echo",
            description="Echo a message back (connectivity test)",
            inputSchema=EchoInput.model_json_schema(),
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "echo":
        inputs = EchoInput.model_validate(arguments)
        return [TextContent(type="text", text=f"Echo: {inputs.message}")]
    raise ValueError(f"Unknown tool: {name}")
```

### Verification Steps
```bash
# 1. Start server manually
python -m gemini_mcp_server

# 2. In Claude Code, run:
/mcp list
# Should show "gemini" server with "echo" tool

# 3. Test tool call:
# Ask Claude: "Use the echo tool to say hello"
```

### Deliverables
- [ ] Minimal MCP server with `echo` tool
- [ ] Server starts without errors
- [ ] `/mcp list` shows the server
- [ ] Tool can be called successfully
- [ ] Response displays correctly in Claude

---

## Milestone 2: Idiomatic Refactor with Streaming

**Goal**: Refactor to idiomatic pydantic-ai with dependency injection, retry logic, and **streaming from day one**

### Tasks
1. Create `GeminiDeps` dataclass for dependency injection
2. Create typed agents with `deps_type` and `output_type`
3. Add retry logic with tenacity
4. **Implement streaming with `run_stream()` and MCP progress notifications**
5. Refactor existing tools to use new patterns
6. Add proper type hints throughout

### Core Dependencies

```python
# gemini_mcp_server/deps.py
from dataclasses import dataclass, field
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@dataclass
class GeminiDeps:
    """Dependencies injected into all Gemini agents."""
    client: genai.Client
    default_model: str = "gemini-2.0-flash"
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "GeminiDeps":
        """Create deps from environment variables."""
        import os
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        return cls(client=genai.Client(api_key=api_key))
```

### Typed Response Models

```python
# gemini_mcp_server/models.py
from pydantic import BaseModel, Field

class TokenUsage(BaseModel):
    """Token usage tracking."""
    input_tokens: int
    output_tokens: int
    total_tokens: int

class GeminiResponse(BaseModel):
    """Standard response from Gemini."""
    content: str
    usage: TokenUsage | None = None

class AskGeminiInput(BaseModel):
    """Input for ask_gemini tool."""
    prompt: str = Field(description="The question or prompt for Gemini")
    model: str | None = Field(default=None, description="Model to use")

class ListModelsInput(BaseModel):
    """Input for list_gemini_models tool."""
    pass
```

### Retry-Enabled Request Helper

```python
# gemini_mcp_server/client.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

class RateLimitError(Exception):
    """Rate limit exceeded."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((RateLimitError, ResourceExhausted)),
)
async def generate_with_retry(
    client: genai.Client,
    model: str,
    prompt: str,
    **kwargs
) -> genai.types.GenerateContentResponse:
    """Generate content with automatic retry on rate limits."""
    try:
        return await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            **kwargs
        )
    except ResourceExhausted as e:
        raise RateLimitError(str(e)) from e
```

### Refactored Server

```python
# gemini_mcp_server/__init__.py
from pydantic_ai import Agent

from .deps import GeminiDeps
from .models import AskGeminiInput, GeminiResponse

# Typed agent with deps
ask_agent: Agent[GeminiDeps, str] = Agent(
    'google-gla:gemini-2.0-flash',
    deps_type=GeminiDeps,
    output_type=str,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "ask_gemini":
        inputs = AskGeminiInput.model_validate(arguments)
        deps = GeminiDeps.from_env()
        result = await ask_agent.run(inputs.prompt, deps=deps)
        return [TextContent(type="text", text=result.output)]
    ...
```

### Streaming Implementation

```python
# gemini_mcp_server/streaming.py
from mcp.server import Server
from mcp.types import TextContent
from pydantic_ai import Agent

async def stream_ask_gemini(
    agent: Agent,
    prompt: str,
    deps: GeminiDeps,
    progress_token: str | None = None,
    session = None,
) -> str:
    """Stream response with optional MCP progress notifications."""
    chunks: list[str] = []

    async with agent.run_stream(prompt, deps=deps) as result:
        async for text in result.stream_text(delta=True):
            chunks.append(text)

            # Send progress notification if client supports it
            if progress_token and session:
                await session.send_progress_notification(
                    progress_token=progress_token,
                    progress=len(chunks),
                    message=text,
                )

    return "".join(chunks)
```

### Deliverables
- [ ] `gemini_mcp_server/deps.py` - GeminiDeps dataclass
- [ ] `gemini_mcp_server/models.py` - All Pydantic models
- [ ] `gemini_mcp_server/client.py` - Retry-enabled request helpers
- [ ] `gemini_mcp_server/streaming.py` - Streaming helper with MCP progress
- [ ] Refactored `__init__.py` using idiomatic patterns
- [ ] `ask_gemini` uses streaming by default
- [ ] Type hints on all public functions
- [ ] Unit tests with `TestModel` mocking
- [ ] All existing tests passing

---

## Milestone 3: Implement `run_code` Tool

**Goal**: Execute Python code via Gemini and return structured results

### Tool Schema
```python
class RunCodeInput(BaseModel):
    """Input for run_code tool."""
    code: str = Field(description="Python code to execute")
    context: str | None = Field(
        default=None,
        description="Optional context about what the code should accomplish"
    )

class CodeExecutionResult(BaseModel):
    """Structured output from code execution."""
    success: bool
    stdout: str | None = None
    stderr: str | None = None
    result: str | None = None  # Final expression value
    error: str | None = None
```

### Implementation Notes
- **Parse response parts**: Gemini returns `executable_code` and `code_execution_result` parts
- **Extract stdout/stderr**: Must parse the structured response, not just text
- **Stateless**: Each call is independent (no variable persistence between calls)
- **Limited libraries**: pandas, numpy, etc. available; no pip install

### Response Parsing
```python
def _parse_code_execution(response) -> CodeExecutionResult:
    """Extract code execution results from Gemini response parts."""
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'executable_code'):
            code = part.executable_code.code
        if hasattr(part, 'code_execution_result'):
            outcome = part.code_execution_result.outcome
            output = part.code_execution_result.output
            return CodeExecutionResult(
                success=(outcome == "OUTCOME_OK"),
                stdout=output,
                ...
            )
```

### Deliverables
- [ ] `RunCodeInput` Pydantic model
- [ ] `CodeExecutionResult` Pydantic model
- [ ] Response part parsing logic
- [ ] `run_code` tool implementation
- [ ] Unit tests with mocked responses
- [ ] Integration test (real API call)

---

## Milestone 4: Implement Deep Research Tools (with Persistence)

**Goal**: Long-running research with SQLite persistence and cancellation

### Tool Schemas
```python
class DeepResearchInput(BaseModel):
    """Input for deep_research tool."""
    query: str = Field(description="Research question or topic")

class DeepResearchOutput(BaseModel):
    """Output from deep_research tool."""
    job_id: str
    status: Literal["started", "running", "completed", "failed", "cancelled"]
    message: str
    estimated_time: str = "2-10 minutes"

class GetResearchStatusInput(BaseModel):
    """Input for get_research_status tool."""
    job_id: str = Field(description="Job ID from deep_research")

class CancelResearchInput(BaseModel):
    """Input for cancel_research tool."""
    job_id: str = Field(description="Job ID to cancel")

class ResearchResult(BaseModel):
    """Research result."""
    job_id: str
    status: str
    progress: str | None = None
    report: str | None = None  # Markdown formatted
    citations: list[Citation] = []
    usage: TokenUsage | None = None
    error: str | None = None
    elapsed_time: float | None = None  # seconds

class Citation(BaseModel):
    """Research citation."""
    title: str
    url: str
    snippet: str | None = None
```

### SQLite Persistence
```python
# jobs.db schema
CREATE TABLE research_jobs (
    job_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    report TEXT,
    citations TEXT,  -- JSON array
    error TEXT,
    token_usage TEXT  -- JSON object
);
```

### Job Manager
```python
class ResearchJobManager:
    """Manages research job persistence and lifecycle."""

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._init_db()

    async def create_job(self, job_id: str, query: str) -> None: ...
    async def get_job(self, job_id: str) -> ResearchJob | None: ...
    async def update_status(self, job_id: str, status: str, **kwargs) -> None: ...
    async def cleanup_stale(self, max_age_minutes: int = 60) -> int: ...
    async def list_active_jobs(self) -> list[ResearchJob]: ...
```

### Server Startup Recovery
```python
async def _recover_pending_jobs():
    """On startup, check status of any pending jobs from before restart."""
    manager = ResearchJobManager()
    pending = await manager.list_active_jobs()
    for job in pending:
        # Poll Gemini API to get current status
        status = await client.interactions.get(job.job_id)
        await manager.update_status(job.job_id, status.status, ...)
```

### Deliverables
- [ ] SQLite schema and migrations
- [ ] `ResearchJobManager` class
- [ ] `deep_research` tool (starts job, persists to DB)
- [ ] `get_research_status` tool (reads from DB, polls API if needed)
- [ ] `cancel_research` tool
- [ ] Startup recovery for pending jobs
- [ ] Stale job cleanup (60 min TTL)
- [ ] Unit tests
- [ ] Integration test

---

## Milestone 5: Polish & Edge Cases

**Goal**: Production hardening and better UX

### Tasks
1. Output formatting (Markdown for reports)
2. Token usage / cost estimation in responses
3. Graceful handling of API unavailability
4. Input validation (max query length, code size limits)
5. Logging for debugging
6. Streaming fallback for non-supporting clients

### Cost Tracking
```python
class CostEstimate(BaseModel):
    """Estimated API cost."""
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float  # Based on current pricing
```

### Deliverables
- [ ] Markdown formatting for research reports
- [ ] Token usage in all responses
- [ ] Cost estimation helper
- [ ] Input validation
- [ ] Structured logging
- [ ] Graceful degradation on API errors

---

## Milestone 6: Documentation & Testing

**Goal**: Production-ready documentation and test coverage

### Tasks
1. Update README with all tools
2. Add usage examples
3. Document the polling pattern clearly
4. 80%+ test coverage
5. CI workflow (GitHub Actions)

### README Updates
```markdown
## Tools

| Tool | Description | Returns |
|------|-------------|---------|
| `ask_gemini` | Quick Q&A | Text response |
| `list_gemini_models` | List available models | Model list |
| `run_code` | Execute Python code | stdout/stderr/result |
| `deep_research` | Start research task | Job ID (async) |
| `get_research_status` | Check research progress | Status/Report |
| `cancel_research` | Cancel running research | Confirmation |

## Deep Research Usage

1. Start research:
   `deep_research(query="History of quantum computing")`
   → Returns `job_id: "abc123"`

2. Poll for results (wait 30-60s between polls):
   `get_research_status(job_id="abc123")`
   → Returns status, progress, or final report

3. Cancel if needed:
   `cancel_research(job_id="abc123")`
```

### Deliverables
- [ ] Updated README.md
- [ ] Usage examples in `examples/` directory
- [ ] pytest coverage ≥80%
- [ ] GitHub Actions CI workflow
- [ ] CHANGELOG.md

---

## Final Tool Summary

| Tool | Description | Sync/Async | API |
|------|-------------|------------|-----|
| `ask_gemini` | Quick Q&A | Sync | generate_content |
| `list_gemini_models` | List models | Sync | models.list |
| `run_code` | Execute Python | Sync | generate_content + code_execution |
| `deep_research` | Start research | Async (returns job_id) | interactions.create |
| `get_research_status` | Poll research | Sync | interactions.get + SQLite |
| `cancel_research` | Cancel research | Sync | interactions.cancel + SQLite |

---

## Dependencies to Add

```toml
[project.dependencies]
# Existing
pydantic-ai = "..."
mcp = "..."
python-dotenv = "..."
google-genai = "..."

# New
tenacity = ">=8.0"  # Retry logic
aiosqlite = ">=0.19"  # Async SQLite
```

---

## Timeline Estimate

| Milestone | Complexity |
|-----------|------------|
| M0: Tooling setup | Low |
| M1: MCP skeleton & connectivity | Low |
| M2: Idiomatic refactor + streaming | Medium |
| M3: run_code | Medium |
| M4: deep_research + SQLite | High |
| M5: Polish & edge cases | Low |
| M6: Docs & tests | Low |

---

## Decisions Made (from Gemini Reviews)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Persist jobs to disk? | **Yes, SQLite** | Server restarts lose in-memory state |
| Allow specifying packages? | **No** | Gemini sandbox doesn't support pip install |
| Rate limiting strategy? | **Per-client with tenacity** | Exponential backoff on 429s |
| Add cancel_research? | **Yes** | Users shouldn't wait 10 min for wrong query |
| Error handling milestone? | **Moved to M2** | Rate limits hit immediately during dev |
| `client` in GeminiDeps? | **Yes, for some tools** | Needed for run_code, deep_research (direct API calls) |
| Add MCP skeleton first? | **Yes (M1)** | Verify connectivity before adding AI complexity |
| Add mypy? | **Yes** | Ruff doesn't do deep type checking |
| Pin pydantic-ai version? | **Yes (`<1.0`)** | Library evolving rapidly |
| Add streaming? | **Yes, core (M2)** | Better UX, implemented from the start |
