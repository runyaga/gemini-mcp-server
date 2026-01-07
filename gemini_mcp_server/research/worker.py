"""Background worker for processing research jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from gemini_mcp_server.models import ResearchState
from gemini_mcp_server.research.engine import execute_research
from gemini_mcp_server.research.store import ResearchStore

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)


class ResearchWorker:
    """Processes research jobs in background."""

    def __init__(
        self,
        store: ResearchStore,
        client: genai.Client,
        poll_interval: float = 2.0,
    ) -> None:
        """Initialize the worker.

        Args:
            store: Job persistence store
            client: Gemini API client
            poll_interval: Interval between polling for new jobs (seconds)
        """
        self.store = store
        self.client = client
        self.poll_interval = poll_interval
        self._running = False
        self._current_tasks: dict[str, asyncio.Task[None]] = {}

    async def process_job(self, job_id: str) -> None:
        """Process a single research job.

        Args:
            job_id: ID of the job to process
        """
        job = await self.store.get_job(job_id)
        if job is None:
            logger.warning(f"Job {job_id} not found")
            return

        if job.state != ResearchState.PENDING:
            logger.warning(f"Job {job_id} is not pending (state: {job.state})")
            return

        # Mark as running
        now = datetime.now()
        await self.store.update_state(
            job_id,
            ResearchState.RUNNING,
            started_at=now,
        )

        try:
            logger.info(f"Starting research for job {job_id}: {job.topic}")

            result = await execute_research(
                client=self.client,
                job_id=job_id,
                topic=job.topic,
                depth=job.depth,
            )

            # Save result and mark complete
            await self.store.save_result(job_id, result)
            await self.store.update_state(
                job_id,
                ResearchState.COMPLETED,
                completed_at=datetime.now(),
            )
            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            await self.store.save_error(job_id, str(e))
            await self.store.update_state(
                job_id,
                ResearchState.FAILED,
                completed_at=datetime.now(),
            )

        finally:
            # Clean up task reference
            self._current_tasks.pop(job_id, None)

    def spawn_job(self, job_id: str) -> asyncio.Task[None]:
        """Spawn a background task to process a job.

        Args:
            job_id: ID of the job to process

        Returns:
            The spawned asyncio Task
        """
        if job_id in self._current_tasks:
            return self._current_tasks[job_id]

        task = asyncio.create_task(self.process_job(job_id))
        self._current_tasks[job_id] = task
        return task

    async def run_loop(self) -> None:
        """Continuous loop for processing pending jobs.

        This is for standalone daemon mode.
        """
        self._running = True
        logger.info("Research worker started")

        try:
            while self._running:
                # Get pending jobs
                pending = await self.store.get_pending_jobs()

                for job in pending:
                    if job.id not in self._current_tasks:
                        self.spawn_job(job.id)

                await asyncio.sleep(self.poll_interval)

        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")
        finally:
            self._running = False
            # Wait for all current tasks to complete
            if self._current_tasks:
                logger.info(f"Waiting for {len(self._current_tasks)} tasks to complete")
                await asyncio.gather(
                    *self._current_tasks.values(), return_exceptions=True
                )
            logger.info("Research worker stopped")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._running = False

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            True if job was cancelled, False if not found/running
        """
        task = self._current_tasks.get(job_id)
        if task is None:
            return False

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        await self.store.update_state(
            job_id,
            ResearchState.FAILED,
            completed_at=datetime.now(),
        )
        await self.store.save_error(job_id, "Cancelled by user")
        return True

    @property
    def is_running(self) -> bool:
        """Check if the worker loop is running."""
        return self._running

    @property
    def active_jobs(self) -> list[str]:
        """Get list of currently processing job IDs."""
        return list(self._current_tasks.keys())
