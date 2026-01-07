"""High-level API for research operations."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from google import genai

from gemini_mcp_server.models import (
    ResearchDepth,
    ResearchJob,
    ResearchState,
    ResearchStatus,
)
from gemini_mcp_server.research.store import ResearchStore
from gemini_mcp_server.research.worker import ResearchWorker

if TYPE_CHECKING:
    pass


class ResearchManager:
    """High-level API for research operations.

    This class provides a simple interface for:
    - Starting research jobs
    - Checking job status
    - Listing jobs
    - Waiting for completion (for sync workflows)

    Example usage (Option B - Python API):
        ```python
        from gemini_mcp_server.research import ResearchManager

        manager = ResearchManager(db_path="research.db")
        job = await manager.start(topic="quantum computing")
        result = await manager.wait_for(job.id, timeout=300)
        print(result.result.summary)
        ```
    """

    def __init__(
        self,
        db_path: str = "research.db",
        client: genai.Client | None = None,
    ) -> None:
        """Initialize the research manager.

        Args:
            db_path: Path to SQLite database for persistence
            client: Gemini API client (created from env if not provided)
        """
        self.store = ResearchStore(db_path=db_path)

        if client is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
            client = genai.Client(api_key=api_key)

        self.client = client
        self.worker = ResearchWorker(store=self.store, client=self.client)

    async def start(
        self,
        topic: str,
        depth: ResearchDepth = ResearchDepth.STANDARD,
    ) -> ResearchJob:
        """Start a research job. Returns immediately with job.

        Args:
            topic: Research topic or question
            depth: How thorough the research should be

        Returns:
            The created ResearchJob (in PENDING state)
        """
        job = await self.store.create_job(topic=topic, depth=depth)

        # Spawn background task to process the job
        self.worker.spawn_job(job.id)

        return job

    async def get_status(self, job_id: str) -> ResearchStatus | None:
        """Get current status of a job.

        Args:
            job_id: Job ID from start()

        Returns:
            ResearchStatus with job info and result/error if complete,
            or None if job not found
        """
        return await self.store.get_status(job_id)

    async def list_jobs(
        self,
        state: ResearchState | None = None,
        limit: int = 10,
    ) -> list[ResearchJob]:
        """List jobs, optionally filtered by state.

        Args:
            state: Filter by state (optional)
            limit: Maximum number of jobs to return (default 10)

        Returns:
            List of ResearchJob objects, ordered by creation time (newest first)
        """
        return await self.store.list_jobs(state=state, limit=limit)

    async def wait_for(
        self,
        job_id: str,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> ResearchStatus:
        """Wait for job completion (for sync workflows).

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds (default 300 = 5 minutes)
            poll_interval: How often to check status (default 2 seconds)

        Returns:
            Final ResearchStatus with result or error

        Raises:
            TimeoutError: If job doesn't complete within timeout
            ValueError: If job not found
        """
        elapsed = 0.0

        while elapsed < timeout:
            status = await self.store.get_status(job_id)

            if status is None:
                raise ValueError(f"Job not found: {job_id}")

            if status.job.is_terminal:
                return status

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled, False if not found or already complete
        """
        status = await self.store.get_status(job_id)
        if status is None:
            return False

        if status.job.is_terminal:
            return False

        # Try to cancel via worker
        if await self.worker.cancel_job(job_id):
            return True

        # If job isn't running yet, just mark it failed
        if status.job.state == ResearchState.PENDING:
            await self.store.update_state(job_id, ResearchState.FAILED)
            await self.store.save_error(job_id, "Cancelled by user")
            return True

        return False

    async def delete(self, job_id: str) -> bool:
        """Delete a job and all associated data.

        Args:
            job_id: Job ID to delete

        Returns:
            True if deleted, False if not found
        """
        return await self.store.delete_job(job_id)

    @property
    def active_jobs(self) -> list[str]:
        """Get list of currently processing job IDs."""
        return self.worker.active_jobs
