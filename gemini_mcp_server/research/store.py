"""SQLite-backed job persistence for research jobs."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from gemini_mcp_server.models import (
    ResearchDepth,
    ResearchJob,
    ResearchResult,
    ResearchState,
    ResearchStatus,
)

SCHEMA = """
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
    sources_json TEXT,
    raw_response TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_errors (
    job_id TEXT PRIMARY KEY REFERENCES research_jobs(id),
    error TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON research_jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON research_jobs(created_at);
"""


class ResearchStore:
    """SQLite-backed job persistence."""

    def __init__(self, db_path: str = "research.db") -> None:
        self.db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _get_conn(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _row_to_job(self, row: sqlite3.Row) -> ResearchJob:
        """Convert a database row to a ResearchJob."""
        return ResearchJob(
            id=row["id"],
            topic=row["topic"],
            depth=ResearchDepth(row["depth"]),
            state=ResearchState(row["state"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(
                datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
        )

    def _row_to_result(self, row: sqlite3.Row) -> ResearchResult:
        """Convert a database row to a ResearchResult."""
        sources = json.loads(row["sources_json"]) if row["sources_json"] else []
        return ResearchResult(
            job_id=row["job_id"],
            summary=row["summary"],
            sources=sources,
            raw_response=row["raw_response"],
        )

    async def create_job(self, topic: str, depth: ResearchDepth) -> ResearchJob:
        """Create a new research job."""
        async with self._lock:
            job_id = str(uuid.uuid4())
            now = datetime.now()

            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO research_jobs (id, topic, depth, state, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        topic,
                        depth.value,
                        ResearchState.PENDING.value,
                        now.isoformat(),
                    ),
                )

            return ResearchJob(
                id=job_id,
                topic=topic,
                depth=depth,
                state=ResearchState.PENDING,
                created_at=now,
            )

    async def get_job(self, job_id: str) -> ResearchJob | None:
        """Get a job by ID."""
        async with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM research_jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()

                if row is None:
                    return None
                return self._row_to_job(row)

    async def update_state(
        self,
        job_id: str,
        state: ResearchState,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        """Update job state."""
        async with self._lock:
            with self._get_conn() as conn:
                updates = ["state = ?"]
                params: list[str | None] = [state.value]

                if started_at is not None:
                    updates.append("started_at = ?")
                    params.append(started_at.isoformat())

                if completed_at is not None:
                    updates.append("completed_at = ?")
                    params.append(completed_at.isoformat())

                params.append(job_id)
                conn.execute(
                    f"UPDATE research_jobs SET {', '.join(updates)} WHERE id = ?",
                    params,
                )

    async def save_result(self, job_id: str, result: ResearchResult) -> None:
        """Save a research result."""
        async with self._lock:
            now = datetime.now()
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO research_results
                    (job_id, summary, sources_json, raw_response, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        result.summary,
                        json.dumps(result.sources),
                        result.raw_response,
                        now.isoformat(),
                    ),
                )

    async def save_error(self, job_id: str, error: str) -> None:
        """Save an error for a job."""
        async with self._lock:
            now = datetime.now()
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO research_errors
                    (job_id, error, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (job_id, error, now.isoformat()),
                )

    async def get_result(self, job_id: str) -> ResearchResult | None:
        """Get the result for a job."""
        async with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM research_results WHERE job_id = ?",
                    (job_id,),
                ).fetchone()

                if row is None:
                    return None
                return self._row_to_result(row)

    async def get_error(self, job_id: str) -> str | None:
        """Get the error for a job."""
        async with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT error FROM research_errors WHERE job_id = ?",
                    (job_id,),
                ).fetchone()

                if row is None:
                    return None
                return str(row["error"])

    async def list_jobs(
        self,
        state: ResearchState | None = None,
        limit: int = 10,
    ) -> list[ResearchJob]:
        """List jobs, optionally filtered by state."""
        async with self._lock:
            with self._get_conn() as conn:
                if state is not None:
                    rows = conn.execute(
                        """
                        SELECT * FROM research_jobs
                        WHERE state = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (state.value, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM research_jobs
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()

                return [self._row_to_job(row) for row in rows]

    async def get_status(self, job_id: str) -> ResearchStatus | None:
        """Get full status for a job including result or error."""
        job = await self.get_job(job_id)
        if job is None:
            return None

        result = None
        error = None

        if job.state == ResearchState.COMPLETED:
            result = await self.get_result(job_id)
        elif job.state == ResearchState.FAILED:
            error = await self.get_error(job_id)

        return ResearchStatus(job=job, result=result, error=error)

    async def get_pending_jobs(self) -> list[ResearchJob]:
        """Get all pending jobs (for worker)."""
        return await self.list_jobs(state=ResearchState.PENDING, limit=100)

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated result/error."""
        async with self._lock:
            with self._get_conn() as conn:
                # Delete from all tables
                conn.execute("DELETE FROM research_results WHERE job_id = ?", (job_id,))
                conn.execute("DELETE FROM research_errors WHERE job_id = ?", (job_id,))
                cursor = conn.execute(
                    "DELETE FROM research_jobs WHERE id = ?", (job_id,)
                )
                return cursor.rowcount > 0
