"""Configuration and security for Gemini MCP Server."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]


# Default deny patterns - always blocked regardless of allowed paths
# These use glob-style patterns where:
#   * matches anything except /
#   ** matches anything including /
#   ? matches single character
DEFAULT_DENY_PATTERNS: list[str] = [
    # Environment files
    ".env",
    ".env.*",
    ".envrc",
    # Credentials and secrets (filename patterns)
    "*credentials*",
    "*secret*",
    "*.pem",
    "*.key",
    "*password*",
    # Directory-based patterns (blocks files inside these dirs)
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".gcloud",
    # Package manager tokens
    ".npmrc",
    ".pypirc",
    # Git credentials
    ".git-credentials",
    ".netrc",
]


def _matches_deny_pattern(resolved_path: Path, pattern: str) -> bool:
    """Check if a path matches a deny pattern.

    Patterns are matched against:
    1. The filename itself
    2. Any directory component in the path (for directory-based blocking)

    Args:
        resolved_path: The resolved absolute path to check
        pattern: The deny pattern to match against

    Returns:
        True if the path should be denied
    """
    # Check filename against pattern
    if fnmatch.fnmatch(resolved_path.name, pattern):
        return True

    # Check if any path component matches (for directory patterns like .ssh, .aws)
    return any(fnmatch.fnmatch(part, pattern) for part in resolved_path.parts)


CONFIG_FILENAME = "config.toml"
CONFIG_DIRS = [
    Path("~/.config/gemini-mcp-server").expanduser(),
    Path("~/.gemini-mcp-server").expanduser(),
]


@dataclass
class ReadFileConfig:
    """Configuration for the read_file tool."""

    enabled: bool = True
    allowed_paths: list[Path] = field(default_factory=list)
    deny_patterns: list[str] = field(
        default_factory=lambda: DEFAULT_DENY_PATTERNS.copy()
    )

    def is_path_allowed(self, file_path: Path) -> tuple[bool, str]:
        """Check if a file path is allowed to be read.

        Args:
            file_path: The path to validate

        Returns:
            Tuple of (is_allowed, error_message)
        """
        if not self.enabled:
            return False, "read_file is disabled in configuration"

        if not self.allowed_paths:
            return False, (
                "read_file not configured. Create ~/.config/gemini-mcp-server/config.toml "
                "with [read_file] allowed = ['~/your/dev/path']"
            )

        # Resolve to real path (follows symlinks, normalizes ..)
        try:
            resolved = file_path.expanduser().resolve(strict=True)
        except FileNotFoundError:
            return False, f"File not found: {file_path}"
        except (OSError, ValueError) as e:
            return False, f"Invalid path: {e}"

        # Check deny patterns first (security takes precedence)
        for pattern in self.deny_patterns:
            if _matches_deny_pattern(resolved, pattern):
                return False, f"Access denied: path matches deny pattern '{pattern}'"

        # Check if path is under any allowed root
        for allowed_root in self.allowed_paths:
            try:
                allowed_resolved = allowed_root.expanduser().resolve()
                if resolved.is_relative_to(allowed_resolved):
                    return True, ""
            except (OSError, ValueError):
                continue

        allowed_str = ", ".join(str(p) for p in self.allowed_paths)
        return (
            False,
            f"Access denied: path not under allowed directories ({allowed_str})",
        )


@dataclass
class ServerConfig:
    """Server-wide configuration."""

    read_file: ReadFileConfig = field(default_factory=ReadFileConfig)


def find_config_file() -> Path | None:
    """Find the configuration file.

    Searches in order:
    1. GEMINI_MCP_CONFIG env var
    2. ~/.config/gemini-mcp-server/config.toml
    3. ~/.gemini-mcp-server/config.toml

    Returns:
        Path to config file if found, None otherwise
    """
    # Check env var first
    env_path = os.environ.get("GEMINI_MCP_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path

    # Check standard locations
    for config_dir in CONFIG_DIRS:
        config_path = config_dir / CONFIG_FILENAME
        if config_path.exists():
            return config_path

    return None


def load_config() -> ServerConfig:
    """Load configuration from file.

    Returns:
        ServerConfig with loaded settings, or defaults if no config found
    """
    config_path = find_config_file()

    if config_path is None:
        return ServerConfig()

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        # Log error but return defaults to not break the server
        import sys

        print(
            f"Warning: Failed to load config from {config_path}: {e}", file=sys.stderr
        )
        return ServerConfig()

    return _parse_config(data)


def _parse_config(data: dict) -> ServerConfig:
    """Parse configuration dictionary into ServerConfig."""
    config = ServerConfig()

    if "read_file" in data:
        rf_data = data["read_file"]

        if "enabled" in rf_data:
            config.read_file.enabled = bool(rf_data["enabled"])

        if "allowed" in rf_data:
            config.read_file.allowed_paths = [
                Path(p).expanduser() for p in rf_data["allowed"]
            ]

        if "deny" in rf_data:
            # Extend default deny patterns with user-specified ones
            config.read_file.deny_patterns.extend(rf_data["deny"])

        if "deny_override" in rf_data:
            # Replace default deny patterns entirely
            config.read_file.deny_patterns = list(rf_data["deny_override"])

    return config


# Singleton config instance (lazy loaded)
_config: ServerConfig | None = None


def get_config() -> ServerConfig:
    """Get the server configuration (singleton).

    Returns:
        ServerConfig instance
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the configuration singleton (for testing)."""
    global _config
    _config = None
