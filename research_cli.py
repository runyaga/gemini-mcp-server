#!/usr/bin/env python
"""CLI for deep research."""

import argparse
import asyncio
import os
import sys

# Ensure API key is set
if not os.environ.get("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable required")
    sys.exit(1)

from gemini_mcp_server.research import ResearchDepth, ResearchManager, ResearchState


async def start_research(topic: str, depth: str) -> None:
    """Start a new research job."""
    manager = ResearchManager()
    depth_enum = ResearchDepth(depth)
    job = await manager.start(topic=topic, depth=depth_enum)
    print("Started research job:")
    print(f"  ID: {job.id}")
    print(f"  Topic: {job.topic}")
    print(f"  Depth: {job.depth.value}")
    print(f"\nCheck status with: python research_cli.py status {job.id}")


async def check_status(job_id: str) -> None:
    """Check status of a research job."""
    manager = ResearchManager()
    status = await manager.get_status(job_id)

    if status is None:
        print(f"Job not found: {job_id}")
        return

    print(f"Job ID: {status.job.id}")
    print(f"Topic: {status.job.topic}")
    print(f"State: {status.job.state.value}")

    if status.result:
        print(f"\n{'=' * 60}")
        print("RESEARCH SUMMARY")
        print(f"{'=' * 60}\n")
        print(status.result.summary)
        if status.result.sources:
            print(f"\n{'=' * 60}")
            print("SOURCES")
            print(f"{'=' * 60}")
            for source in status.result.sources:
                print(f"  - {source}")

    if status.error:
        print(f"\nError: {status.error}")


async def wait_for_result(job_id: str, timeout: int) -> None:
    """Wait for a research job to complete."""
    manager = ResearchManager()
    print(f"Waiting for job {job_id} (timeout: {timeout}s)...")

    try:
        await manager.wait_for(job_id, timeout=timeout)
        await check_status(job_id)
    except TimeoutError:
        print(f"Job did not complete within {timeout} seconds")
    except ValueError as e:
        print(f"Error: {e}")


async def list_jobs(state: str | None, limit: int) -> None:
    """List research jobs."""
    manager = ResearchManager()
    state_enum = ResearchState(state) if state else None
    jobs = await manager.list_jobs(state=state_enum, limit=limit)

    if not jobs:
        print("No jobs found")
        return

    print(f"{'ID':<36} {'State':<10} {'Topic'}")
    print("-" * 80)
    for job in jobs:
        print(f"{job.id} {job.state.value:<10} {job.topic[:40]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start command
    start = subparsers.add_parser("start", help="Start a research job")
    start.add_argument("topic", help="Research topic")
    start.add_argument(
        "--depth",
        "-d",
        choices=["quick", "standard", "thorough"],
        default="standard",
        help="Research depth (default: standard)",
    )

    # status command
    status = subparsers.add_parser("status", help="Check job status")
    status.add_argument("job_id", help="Job ID")

    # wait command
    wait = subparsers.add_parser("wait", help="Wait for job completion")
    wait.add_argument("job_id", help="Job ID")
    wait.add_argument(
        "--timeout", "-t", type=int, default=300, help="Timeout in seconds"
    )

    # list command
    lst = subparsers.add_parser("list", help="List jobs")
    lst.add_argument(
        "--state",
        "-s",
        choices=["pending", "running", "completed", "failed"],
        help="Filter by state",
    )
    lst.add_argument("--limit", "-n", type=int, default=10, help="Max results")

    args = parser.parse_args()

    if args.command == "start":
        asyncio.run(start_research(args.topic, args.depth))
    elif args.command == "status":
        asyncio.run(check_status(args.job_id))
    elif args.command == "wait":
        asyncio.run(wait_for_result(args.job_id, args.timeout))
    elif args.command == "list":
        asyncio.run(list_jobs(args.state, args.limit))


if __name__ == "__main__":
    main()
