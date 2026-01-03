# Deep Research Implementation Plan

## Overview

Add async deep research capability that works for both:
- **Option A**: Claude Code via MCP tools (polling-based)
- **Option B**: pydantic-ai agent loops (direct Python API)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     gemini-mcp-server                       │
├─────────────────────────────────────────────────────────────┤
│  MCP Tools (Option A)                                       │
│  ┌─────────────────┐ ┌──────────────────┐ ┌───────────────┐ │
│  │ start_research  │ │ get_research     │ │ list_research │ │
│  │   → job_id      │ │   → status/result│ │   → jobs[]    │ │
│  └─────────────────┘ └──────────────────┘ └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Python API (Option B)                                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ from gemini_mcp_server.research import ResearchManager  ││
│  │                                                         ││
│  │ manager = ResearchManager(db_path="research.db")        ││
│  │ job = await manager.start(topic="quantum computing")    ││
│  │ result = await manager.wait_for(job.id, timeout=300)    ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Research Engine                                            │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────────┐│
│  │ SQLite Job │  │ Background │  │ Gemini Deep Research    ││
│  │   Store    │◄─│   Worker   │◄─│ (grounding + search)    ││
│  └────────────┘  └────────────┘  └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Models

### Enums

```python
class ResearchState(str, Enum):
    """State of a research job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

### Pydantic Models

```python
class ResearchJob(BaseModel):
    """A background research job."""
    id: str = Field(description="Unique job identifier")
    topic: str = Field(description="Research topic")
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
```

### MCP Tool Inputs

```python
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
```

## SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS research_jobs (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    depth TEXT NOT NULL DEFAULT 'standard',
    state TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS research_results (
    job_id TEXT PRIMARY KEY REFERENCES research_jobs(id),
    summary TEXT NOT NULL,
    sources_json TEXT,  -- JSON array of source URLs
    raw_response TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_errors (
    job_id TEXT PRIMARY KEY REFERENCES research_jobs(id),
    error TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_jobs_state ON research_jobs(state);
CREATE INDEX idx_jobs_created ON research_jobs(created_at);
```

## File Structure

```
gemini_mcp_server/
├── __init__.py          # Add MCP tool handlers
├── models.py            # Add research models
├── research/
│   ├── __init__.py      # Public API exports
│   ├── manager.py       # ResearchManager class
│   ├── store.py         # SQLite job store
│   ├── worker.py        # Background worker
│   └── engine.py        # Gemini research execution
```

## Implementation Steps

### Step 1: Models (`models.py`)

Add to existing models file:
- `ResearchState` enum
- `ResearchDepth` enum (QUICK, STANDARD, THOROUGH)
- `ResearchJob`, `ResearchResult`, `ResearchStatus` models
- MCP input models

### Step 2: SQLite Store (`research/store.py`)

```python
class ResearchStore:
    """SQLite-backed job persistence."""

    def __init__(self, db_path: str = "research.db") -> None: ...
    async def create_job(self, topic: str, depth: ResearchDepth) -> ResearchJob: ...
    async def get_job(self, job_id: str) -> ResearchJob | None: ...
    async def update_state(self, job_id: str, state: ResearchState) -> None: ...
    async def save_result(self, job_id: str, result: ResearchResult) -> None: ...
    async def save_error(self, job_id: str, error: str) -> None: ...
    async def list_jobs(
        self,
        state: ResearchState | None = None,
        limit: int = 10,
    ) -> list[ResearchJob]: ...
    async def get_status(self, job_id: str) -> ResearchStatus: ...
```

### Step 3: Research Engine (`research/engine.py`)

```python
async def execute_research(
    client: genai.Client,
    topic: str,
    depth: ResearchDepth,
) -> ResearchResult:
    """Execute deep research using Gemini with grounding."""
    # Use Gemini's grounding/search capabilities
    # Parse and structure the response
    # Extract sources
```

### Step 4: Background Worker (`research/worker.py`)

```python
class ResearchWorker:
    """Processes research jobs in background."""

    def __init__(self, store: ResearchStore, client: genai.Client) -> None: ...

    async def process_job(self, job_id: str) -> None:
        """Process a single job (called by manager)."""
        ...

    async def run_loop(self) -> None:
        """Continuous loop for processing pending jobs."""
        # For standalone daemon mode
        ...
```

### Step 5: Manager (`research/manager.py`)

```python
class ResearchManager:
    """High-level API for research operations."""

    def __init__(
        self,
        db_path: str = "research.db",
        client: genai.Client | None = None,
    ) -> None: ...

    async def start(
        self,
        topic: str,
        depth: ResearchDepth = ResearchDepth.STANDARD,
    ) -> ResearchJob:
        """Start a research job. Returns immediately with job."""
        ...

    async def get_status(self, job_id: str) -> ResearchStatus:
        """Get current status of a job."""
        ...

    async def list_jobs(
        self,
        state: ResearchState | None = None,
        limit: int = 10,
    ) -> list[ResearchJob]:
        """List jobs, optionally filtered by state."""
        ...

    async def wait_for(
        self,
        job_id: str,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> ResearchStatus:
        """Wait for job completion (for sync workflows)."""
        ...

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending/running job."""
        ...
```

### Step 6: MCP Tool Handlers (`__init__.py`)

Add three new tools:
- `start_research` - creates job, spawns background task
- `get_research` - returns status/result
- `list_research` - lists jobs with optional filter

### Step 7: Tests

- Unit tests for store operations
- Unit tests for manager
- Integration test with real Gemini (marked)
- Mock tests for MCP handlers

## Option B Integration Example

For use in `~/dev/agentic-design`:

```python
from dataclasses import dataclass, field
from gemini_mcp_server.research import ResearchManager, ResearchState
from pydantic_ai import Agent, RunContext


@dataclass
class ResearchAwareDeps:
    """Dependencies with research capability."""
    research: ResearchManager
    pending_jobs: list[str] = field(default_factory=list)


agent = Agent(
    model,
    deps_type=ResearchAwareDeps,
    system_prompt="You are a research assistant.",
)


@agent.system_prompt
async def inject_completed_research(ctx: RunContext[ResearchAwareDeps]) -> str:
    """Check for completed research between turns."""
    completed = []
    still_pending = []

    for job_id in ctx.deps.pending_jobs:
        status = await ctx.deps.research.get_status(job_id)
        if status.job.is_complete:
            completed.append(
                f"Research on '{status.job.topic}' complete:\n"
                f"{status.result.summary}"
            )
        elif status.job.is_terminal:
            completed.append(
                f"Research on '{status.job.topic}' failed: {status.error}"
            )
        else:
            still_pending.append(job_id)

    ctx.deps.pending_jobs[:] = still_pending

    if completed:
        return "BACKGROUND RESEARCH UPDATES:\n\n" + "\n\n---\n\n".join(completed)
    return ""


@agent.tool
async def start_background_research(
    ctx: RunContext[ResearchAwareDeps],
    topic: str,
) -> str:
    """Start background research on a topic."""
    job = await ctx.deps.research.start(topic)
    ctx.deps.pending_jobs.append(job.id)
    return f"Started research job {job.id} on '{topic}'"


@agent.tool
async def check_research_status(
    ctx: RunContext[ResearchAwareDeps],
) -> str:
    """Check status of all pending research."""
    if not ctx.deps.pending_jobs:
        return "No pending research jobs."

    lines = []
    for job_id in ctx.deps.pending_jobs:
        status = await ctx.deps.research.get_status(job_id)
        lines.append(f"- {job_id}: {status.job.state.value} ({status.job.topic})")

    return "Pending research:\n" + "\n".join(lines)
```

## Open Questions

1. **Background execution model**:
   - asyncio.create_task (in-process, dies with server)?
   - Separate worker process?
   - For MCP, likely needs to be in-process task

2. **Job expiration**:
   - Auto-delete old completed jobs?
   - TTL on jobs (e.g., 24 hours)?

3. **Gemini grounding API**:
   - Need to verify exact API for search/grounding
   - May need specific model (gemini-2.0-flash with grounding?)

4. **Rate limiting**:
   - Queue depth limits?
   - Concurrent job limits?

## Success Criteria

- [ ] `start_research` returns immediately with job_id
- [ ] Research executes in background
- [ ] `get_research` returns status without blocking
- [ ] `list_research` shows all jobs with state filter
- [ ] Jobs persist across server restarts (SQLite)
- [ ] Python API works standalone (no MCP needed)
- [ ] Integration with agentic-design deps pattern works
- [ ] Tests pass with 80%+ coverage on new code
