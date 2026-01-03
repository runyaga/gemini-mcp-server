"""Pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module imports between tests."""
    import sys

    # Remove cached imports to allow fresh mocking
    modules_to_remove = [k for k in sys.modules if k.startswith("gemini_mcp_server")]
    for mod in modules_to_remove:
        del sys.modules[mod]
