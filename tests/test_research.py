"""Tests for research functionality."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemini_mcp_server.models import (
    ResearchDepth,
    ResearchJob,
    ResearchResult,
    ResearchState,
    ResearchStatus,
)


class TestResearchModels:
    """Test research Pydantic models."""

    def test_research_state_enum(self):
        """Should define expected states."""
        assert ResearchState.PENDING.value == "pending"
        assert ResearchState.RUNNING.value == "running"
        assert ResearchState.COMPLETED.value == "completed"
        assert ResearchState.FAILED.value == "failed"

    def test_research_depth_enum(self):
        """Should define expected depths."""
        assert ResearchDepth.QUICK.value == "quick"
        assert ResearchDepth.STANDARD.value == "standard"
        assert ResearchDepth.THOROUGH.value == "thorough"

    def test_research_job_creation(self):
        """Should create job with defaults."""
        job = ResearchJob(id="test-id", topic="test topic")
        assert job.id == "test-id"
        assert job.topic == "test topic"
        assert job.depth == ResearchDepth.STANDARD
        assert job.state == ResearchState.PENDING
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None

    def test_research_job_is_complete(self):
        """Should correctly identify complete jobs."""
        job = ResearchJob(id="1", topic="test", state=ResearchState.PENDING)
        assert job.is_complete is False

        job = ResearchJob(id="1", topic="test", state=ResearchState.RUNNING)
        assert job.is_complete is False

        job = ResearchJob(id="1", topic="test", state=ResearchState.COMPLETED)
        assert job.is_complete is True

        job = ResearchJob(id="1", topic="test", state=ResearchState.FAILED)
        assert job.is_complete is False

    def test_research_job_is_terminal(self):
        """Should correctly identify terminal jobs."""
        job = ResearchJob(id="1", topic="test", state=ResearchState.PENDING)
        assert job.is_terminal is False

        job = ResearchJob(id="1", topic="test", state=ResearchState.RUNNING)
        assert job.is_terminal is False

        job = ResearchJob(id="1", topic="test", state=ResearchState.COMPLETED)
        assert job.is_terminal is True

        job = ResearchJob(id="1", topic="test", state=ResearchState.FAILED)
        assert job.is_terminal is True

    def test_research_result(self):
        """Should create result with sources."""
        result = ResearchResult(
            job_id="test-id",
            summary="Test summary",
            sources=["https://example.com"],
            raw_response="Raw response text",
        )
        assert result.job_id == "test-id"
        assert result.summary == "Test summary"
        assert result.sources == ["https://example.com"]
        assert result.raw_response == "Raw response text"

    def test_research_status(self):
        """Should create status with job and result."""
        job = ResearchJob(id="test-id", topic="test")
        result = ResearchResult(job_id="test-id", summary="Summary")
        status = ResearchStatus(job=job, result=result)

        assert status.job == job
        assert status.result == result
        assert status.error is None


class TestResearchStore:
    """Test SQLite research store."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a store with temp database."""
        from gemini_mcp_server.research.store import ResearchStore

        db_path = tmp_path / "test_research.db"
        return ResearchStore(db_path=str(db_path))

    @pytest.mark.asyncio
    async def test_create_job(self, store):
        """Should create and persist a job."""
        job = await store.create_job(topic="Test topic", depth=ResearchDepth.STANDARD)

        assert job.id is not None
        assert len(job.id) == 36  # UUID format
        assert job.topic == "Test topic"
        assert job.depth == ResearchDepth.STANDARD
        assert job.state == ResearchState.PENDING

    @pytest.mark.asyncio
    async def test_get_job(self, store):
        """Should retrieve a job by ID."""
        created = await store.create_job(topic="Test topic", depth=ResearchDepth.QUICK)
        retrieved = await store.get_job(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.topic == created.topic

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, store):
        """Should return None for non-existent job."""
        result = await store.get_job("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_state(self, store):
        """Should update job state."""
        job = await store.create_job(topic="Test", depth=ResearchDepth.STANDARD)
        now = datetime.now()

        await store.update_state(
            job.id,
            ResearchState.RUNNING,
            started_at=now,
        )

        updated = await store.get_job(job.id)
        assert updated.state == ResearchState.RUNNING
        assert updated.started_at is not None

    @pytest.mark.asyncio
    async def test_save_and_get_result(self, store):
        """Should save and retrieve results."""
        job = await store.create_job(topic="Test", depth=ResearchDepth.STANDARD)
        result = ResearchResult(
            job_id=job.id,
            summary="Test summary",
            sources=["https://example.com"],
        )

        await store.save_result(job.id, result)
        retrieved = await store.get_result(job.id)

        assert retrieved is not None
        assert retrieved.summary == "Test summary"
        assert retrieved.sources == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_save_and_get_error(self, store):
        """Should save and retrieve errors."""
        job = await store.create_job(topic="Test", depth=ResearchDepth.STANDARD)

        await store.save_error(job.id, "Test error message")
        error = await store.get_error(job.id)

        assert error == "Test error message"

    @pytest.mark.asyncio
    async def test_list_jobs(self, store):
        """Should list jobs ordered by creation time."""
        await store.create_job(topic="First", depth=ResearchDepth.STANDARD)
        await asyncio.sleep(0.01)
        await store.create_job(topic="Second", depth=ResearchDepth.STANDARD)

        jobs = await store.list_jobs()

        assert len(jobs) == 2
        assert jobs[0].topic == "Second"  # Newest first
        assert jobs[1].topic == "First"

    @pytest.mark.asyncio
    async def test_list_jobs_filtered_by_state(self, store):
        """Should filter jobs by state."""
        await store.create_job(topic="Pending", depth=ResearchDepth.STANDARD)
        job2 = await store.create_job(topic="Running", depth=ResearchDepth.STANDARD)
        await store.update_state(job2.id, ResearchState.RUNNING)

        pending = await store.list_jobs(state=ResearchState.PENDING)
        running = await store.list_jobs(state=ResearchState.RUNNING)

        assert len(pending) == 1
        assert pending[0].topic == "Pending"
        assert len(running) == 1
        assert running[0].topic == "Running"

    @pytest.mark.asyncio
    async def test_get_status(self, store):
        """Should return full status with result or error."""
        job = await store.create_job(topic="Test", depth=ResearchDepth.STANDARD)
        result = ResearchResult(job_id=job.id, summary="Summary")

        await store.update_state(job.id, ResearchState.COMPLETED)
        await store.save_result(job.id, result)

        status = await store.get_status(job.id)

        assert status is not None
        assert status.job.state == ResearchState.COMPLETED
        assert status.result is not None
        assert status.result.summary == "Summary"

    @pytest.mark.asyncio
    async def test_delete_job(self, store):
        """Should delete job and associated data."""
        job = await store.create_job(topic="Test", depth=ResearchDepth.STANDARD)
        result = ResearchResult(job_id=job.id, summary="Summary")
        await store.save_result(job.id, result)

        deleted = await store.delete_job(job.id)
        assert deleted is True

        retrieved = await store.get_job(job.id)
        assert retrieved is None


class TestResearchEngine:
    """Test research execution engine."""

    @pytest.mark.asyncio
    async def test_execute_research_success(self):
        """Should execute research and return result."""
        from gemini_mcp_server.research.engine import execute_research

        # Mock response
        mock_part = MagicMock()
        mock_part.text = "Research summary about quantum computing."

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.grounding_metadata = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await execute_research(
            client=mock_client,
            job_id="test-job",
            topic="quantum computing",
            depth=ResearchDepth.STANDARD,
        )

        assert result.job_id == "test-job"
        assert "quantum computing" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_execute_research_with_sources(self):
        """Should extract sources from grounding metadata."""
        from gemini_mcp_server.research.engine import execute_research

        # Mock grounding chunks
        mock_web_chunk = MagicMock()
        mock_web_chunk.uri = "https://example.com/source1"

        mock_grounding_chunk = MagicMock()
        mock_grounding_chunk.web = mock_web_chunk

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.grounding_chunks = [mock_grounding_chunk]
        mock_grounding_metadata.search_entry_point = None

        mock_part = MagicMock()
        mock_part.text = "Research summary."

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.grounding_metadata = mock_grounding_metadata

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await execute_research(
            client=mock_client,
            job_id="test-job",
            topic="test topic",
            depth=ResearchDepth.QUICK,
        )

        assert "https://example.com/source1" in result.sources


class TestResearchWorker:
    """Test background research worker."""

    @pytest.fixture
    def worker_setup(self, tmp_path):
        """Create store, client, and worker."""
        from gemini_mcp_server.research.store import ResearchStore
        from gemini_mcp_server.research.worker import ResearchWorker

        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        client = MagicMock()
        worker = ResearchWorker(store=store, client=client)
        return store, client, worker

    @pytest.mark.asyncio
    async def test_process_job_success(self, worker_setup):
        """Should process job and save result."""
        store, client, worker = worker_setup

        # Create job
        job = await store.create_job(topic="Test topic", depth=ResearchDepth.QUICK)

        # Mock research execution
        mock_part = MagicMock()
        mock_part.text = "Research result"

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.grounding_metadata = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Process
        await worker.process_job(job.id)

        # Verify
        status = await store.get_status(job.id)
        assert status.job.state == ResearchState.COMPLETED
        assert status.result is not None

    @pytest.mark.asyncio
    async def test_process_job_failure(self, worker_setup):
        """Should handle failure and save error."""
        store, client, worker = worker_setup

        job = await store.create_job(topic="Test", depth=ResearchDepth.STANDARD)
        client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API error")
        )

        await worker.process_job(job.id)

        status = await store.get_status(job.id)
        assert status.job.state == ResearchState.FAILED
        assert status.error is not None
        assert "API error" in status.error

    @pytest.mark.asyncio
    async def test_spawn_job(self, worker_setup):
        """Should spawn background task."""
        store, client, worker = worker_setup

        job = await store.create_job(topic="Test", depth=ResearchDepth.STANDARD)

        # Mock successful execution
        mock_part = MagicMock()
        mock_part.text = "Result"
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.grounding_metadata = None
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        task = worker.spawn_job(job.id)
        assert task is not None
        assert job.id in worker.active_jobs

        await task  # Wait for completion
        assert job.id not in worker.active_jobs


class TestResearchManager:
    """Test high-level research manager."""

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        """Create manager with mocked client."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("gemini_mcp_server.research.manager.genai.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            from gemini_mcp_server.research.manager import ResearchManager

            mgr = ResearchManager(db_path=str(tmp_path / "test.db"))
            mgr._mock_client = mock_client  # Store for test access
            return mgr

    @pytest.mark.asyncio
    async def test_start(self, manager):
        """Should start job and return immediately."""
        job = await manager.start(topic="Test topic")

        assert job.id is not None
        assert job.topic == "Test topic"
        assert job.state == ResearchState.PENDING

    @pytest.mark.asyncio
    async def test_get_status(self, manager):
        """Should return job status."""
        job = await manager.start(topic="Test topic")
        status = await manager.get_status(job.id)

        assert status is not None
        assert status.job.id == job.id

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, manager):
        """Should return None for non-existent job."""
        status = await manager.get_status("non-existent")
        assert status is None

    @pytest.mark.asyncio
    async def test_list_jobs(self, manager):
        """Should list all jobs."""
        await manager.start(topic="Job 1")
        await manager.start(topic="Job 2")

        jobs = await manager.list_jobs()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self, manager):
        """Should raise timeout if job doesn't complete."""
        # Create a job directly in the store to bypass worker
        job = await manager.store.create_job(
            topic="Slow job", depth=ResearchDepth.STANDARD
        )
        # Mark it as running so it doesn't trigger worker and stays non-terminal
        await manager.store.update_state(job.id, ResearchState.RUNNING)

        with pytest.raises(TimeoutError):
            await manager.wait_for(job.id, timeout=0.1, poll_interval=0.05)

    @pytest.mark.asyncio
    async def test_delete(self, manager):
        """Should delete job."""
        job = await manager.start(topic="Test")

        deleted = await manager.delete(job.id)
        assert deleted is True

        status = await manager.get_status(job.id)
        assert status is None


class TestMCPToolHandlers:
    """Test MCP tool handlers for research."""

    @pytest.fixture
    def mock_env(self, monkeypatch, tmp_path):
        """Set up test environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        monkeypatch.setenv("GEMINI_RESEARCH_DB", str(tmp_path / "research.db"))

    @pytest.mark.asyncio
    async def test_list_tools_includes_research(self, mock_env):
        """Should include research tools in list."""
        with patch("gemini_mcp_server.GoogleModel"):
            from gemini_mcp_server import list_tools

            tools = await list_tools()
            tool_names = [t.name for t in tools]

            assert "start_research" in tool_names
            assert "get_research" in tool_names
            assert "list_research" in tool_names

    @pytest.mark.asyncio
    async def test_start_research_tool(self, mock_env):
        """Should start research job via tool."""
        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.research.manager.genai.Client"),
        ):
            # Clear any cached manager
            import gemini_mcp_server

            gemini_mcp_server._research_manager = None

            from gemini_mcp_server import call_tool

            result = await call_tool(
                "start_research",
                {"topic": "quantum computing", "depth": "standard"},
            )

            assert "Research job started" in result[0].text
            assert "Job ID:" in result[0].text
            assert "quantum computing" in result[0].text

    @pytest.mark.asyncio
    async def test_get_research_tool_not_found(self, mock_env):
        """Should handle non-existent job."""
        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.research.manager.genai.Client"),
        ):
            import gemini_mcp_server

            gemini_mcp_server._research_manager = None

            from gemini_mcp_server import call_tool

            result = await call_tool(
                "get_research",
                {"job_id": "non-existent-id"},
            )

            assert "Job not found" in result[0].text

    @pytest.mark.asyncio
    async def test_list_research_tool_empty(self, mock_env):
        """Should handle empty job list."""
        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.research.manager.genai.Client"),
        ):
            import gemini_mcp_server

            gemini_mcp_server._research_manager = None

            from gemini_mcp_server import call_tool

            result = await call_tool("list_research", {})

            assert "No research jobs found" in result[0].text

    @pytest.mark.asyncio
    async def test_list_research_with_jobs(self, mock_env):
        """Should list existing jobs."""
        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.research.manager.genai.Client"),
        ):
            import gemini_mcp_server

            gemini_mcp_server._research_manager = None

            from gemini_mcp_server import call_tool

            # Create a job first
            await call_tool("start_research", {"topic": "test topic"})

            # List jobs
            result = await call_tool("list_research", {})

            assert "Found 1 job(s)" in result[0].text
            assert "test topic" in result[0].text


class TestInputModels:
    """Test research input model validation."""

    def test_start_research_input(self):
        """Should validate start_research input."""
        from gemini_mcp_server.models import StartResearchInput

        inputs = StartResearchInput(topic="Test topic")
        assert inputs.topic == "Test topic"
        assert inputs.depth == ResearchDepth.STANDARD

        inputs = StartResearchInput(topic="Test", depth=ResearchDepth.THOROUGH)
        assert inputs.depth == ResearchDepth.THOROUGH

    def test_get_research_input(self):
        """Should validate get_research input."""
        from gemini_mcp_server.models import GetResearchInput

        inputs = GetResearchInput(job_id="test-id")
        assert inputs.job_id == "test-id"

    def test_list_research_input(self):
        """Should validate list_research input."""
        from gemini_mcp_server.models import ListResearchInput

        inputs = ListResearchInput()
        assert inputs.state is None
        assert inputs.limit == 10

        inputs = ListResearchInput(state=ResearchState.COMPLETED, limit=50)
        assert inputs.state == ResearchState.COMPLETED
        assert inputs.limit == 50

    def test_list_research_input_limit_bounds(self):
        """Should enforce limit bounds."""
        from pydantic import ValidationError

        from gemini_mcp_server.models import ListResearchInput

        with pytest.raises(ValidationError):
            ListResearchInput(limit=0)

        with pytest.raises(ValidationError):
            ListResearchInput(limit=101)
