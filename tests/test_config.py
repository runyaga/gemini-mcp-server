"""Tests for configuration and security."""

import os
from pathlib import Path
from unittest.mock import patch

from gemini_mcp_server.config import (
    DEFAULT_DENY_PATTERNS,
    MAX_WRITE_SIZE_BYTES,
    ReadFileConfig,
    WriteFileConfig,
    _parse_config,
    find_config_file,
    load_config,
    reset_config,
)


class TestReadFileConfig:
    """Test ReadFileConfig path validation."""

    def test_no_allowed_paths_returns_error(self):
        """Should return error when no allowed paths configured."""
        config = ReadFileConfig(allowed_paths=[])
        allowed, msg = config.is_path_allowed(Path("/some/file.txt"))
        assert not allowed
        assert "not configured" in msg

    def test_disabled_returns_error(self):
        """Should return error when read_file is disabled."""
        config = ReadFileConfig(enabled=False, allowed_paths=[Path("/tmp")])
        allowed, msg = config.is_path_allowed(Path("/tmp/file.txt"))
        assert not allowed
        assert "disabled" in msg

    def test_path_under_allowed_dir_succeeds(self, tmp_path):
        """Should allow path under configured allowed directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, _ = config.is_path_allowed(test_file)
        assert allowed

    def test_path_outside_allowed_dir_fails(self, tmp_path):
        """Should deny path outside allowed directories."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        other_dir = tmp_path.parent / "other"
        config = ReadFileConfig(allowed_paths=[other_dir])
        allowed, msg = config.is_path_allowed(test_file)
        assert not allowed
        assert "not under allowed directories" in msg

    def test_nested_path_allowed(self, tmp_path):
        """Should allow deeply nested paths under allowed directory."""
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        test_file = nested / "file.txt"
        test_file.write_text("content")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, _ = config.is_path_allowed(test_file)
        assert allowed

    def test_nonexistent_file_fails(self, tmp_path):
        """Should deny nonexistent files."""
        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, msg = config.is_path_allowed(tmp_path / "nonexistent.txt")
        assert not allowed
        assert "File not found" in msg

    def test_symlink_to_allowed_file_works(self, tmp_path):
        """Should allow symlinks pointing to allowed locations."""
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, _ = config.is_path_allowed(link)
        assert allowed

    def test_symlink_escape_blocked(self, tmp_path):
        """Should block symlinks that escape allowed directory."""
        # Create a file outside allowed directory
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        target_file = outside_dir / "data.txt"
        target_file.write_text("outside content")

        # Create allowed directory with symlink escaping to outside
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        escape_link = allowed_dir / "escape.txt"
        escape_link.symlink_to(target_file)

        # Only allow the 'allowed' subdirectory
        config = ReadFileConfig(allowed_paths=[allowed_dir])
        allowed, msg = config.is_path_allowed(escape_link)
        # The symlink resolves to outside_dir which is not under allowed_dir
        assert not allowed
        assert "not under allowed directories" in msg


class TestDenyPatterns:
    """Test deny pattern functionality."""

    def test_env_file_blocked(self, tmp_path):
        """Should block .env files."""
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, msg = config.is_path_allowed(env_file)
        assert not allowed
        assert "deny pattern" in msg

    def test_env_local_blocked(self, tmp_path):
        """Should block .env.local and similar files."""
        env_file = tmp_path / ".env.local"
        env_file.write_text("SECRET=value")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, msg = config.is_path_allowed(env_file)
        assert not allowed
        assert "deny pattern" in msg

    def test_ssh_key_blocked(self, tmp_path):
        """Should block SSH keys."""
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        key_file = ssh_dir / "id_rsa"
        key_file.write_text("-----BEGIN RSA PRIVATE KEY-----")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, msg = config.is_path_allowed(key_file)
        assert not allowed
        assert "deny pattern" in msg

    def test_aws_credentials_blocked(self, tmp_path):
        """Should block AWS credentials."""
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        creds = aws_dir / "credentials"
        creds.write_text("[default]\naws_access_key_id=...")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, msg = config.is_path_allowed(creds)
        assert not allowed
        assert "deny pattern" in msg

    def test_pem_file_blocked(self, tmp_path):
        """Should block .pem files."""
        pem_file = tmp_path / "server.pem"
        pem_file.write_text("-----BEGIN CERTIFICATE-----")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, msg = config.is_path_allowed(pem_file)
        assert not allowed
        assert "deny pattern" in msg

    def test_normal_file_allowed(self, tmp_path):
        """Should allow normal files not matching deny patterns."""
        code_file = tmp_path / "main.py"
        code_file.write_text("print('hello')")

        config = ReadFileConfig(allowed_paths=[tmp_path])
        allowed, _ = config.is_path_allowed(code_file)
        assert allowed

    def test_custom_deny_pattern(self, tmp_path):
        """Should support custom deny patterns."""
        custom_file = tmp_path / "blocked_file.txt"
        custom_file.write_text("custom content")

        config = ReadFileConfig(
            allowed_paths=[tmp_path],
            deny_patterns=["blocked_file*"],
        )
        allowed, _ = config.is_path_allowed(custom_file)
        assert not allowed


class TestConfigLoading:
    """Test configuration file loading."""

    def test_find_config_from_env(self, tmp_path):
        """Should find config from GEMINI_MCP_CONFIG env var."""
        config_file = tmp_path / "my_config.toml"
        config_file.write_text("[read_file]\nenabled = true\n")

        with patch.dict(os.environ, {"GEMINI_MCP_CONFIG": str(config_file)}):
            found = find_config_file()
            assert found == config_file

    def test_find_config_default_location(self, tmp_path, monkeypatch):
        """Should find config in default location when env var points to it."""
        # Use the env var approach to test config file discovery
        config_dir = tmp_path / ".config" / "gemini-mcp-server"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("[read_file]\nenabled = true\nallowed = ['/tmp']\n")

        # Set env var to the config file
        monkeypatch.setenv("GEMINI_MCP_CONFIG", str(config_file))

        found = find_config_file()
        assert found == config_file

        # Verify the config actually loads correctly
        reset_config()
        config = load_config()
        assert config.read_file.enabled is True
        assert len(config.read_file.allowed_paths) == 1

    def test_no_config_returns_none(self):
        """Should return None when no config file exists."""
        import gemini_mcp_server.config as config_module

        # Patch both env var and CONFIG_DIRS to ensure no config is found
        with (
            patch.dict(os.environ, {"GEMINI_MCP_CONFIG": ""}, clear=False),
            patch.object(config_module, "CONFIG_DIRS", [Path("/nonexistent/path")]),
        ):
            # Need to re-import to get the patched version
            from gemini_mcp_server.config import find_config_file as find_config

            found = find_config()
            assert found is None


class TestParseConfig:
    """Test configuration parsing."""

    def test_parse_empty_config(self):
        """Should return defaults for empty config."""
        config = _parse_config({})
        assert config.read_file.enabled is True
        assert config.read_file.allowed_paths == []
        assert config.read_file.deny_patterns == DEFAULT_DENY_PATTERNS

    def test_parse_allowed_paths(self):
        """Should parse allowed paths."""
        config = _parse_config(
            {"read_file": {"allowed": ["~/dev", "/tmp", "./relative"]}}
        )
        assert len(config.read_file.allowed_paths) == 3
        assert Path("~/dev").expanduser() in config.read_file.allowed_paths

    def test_parse_disabled(self):
        """Should parse enabled=false."""
        config = _parse_config({"read_file": {"enabled": False}})
        assert config.read_file.enabled is False

    def test_parse_additional_deny_patterns(self):
        """Should extend default deny patterns with custom ones."""
        config = _parse_config(
            {"read_file": {"allowed": ["/tmp"], "deny": ["custom_blocked*"]}}
        )
        assert "custom_blocked*" in config.read_file.deny_patterns
        # Default patterns should still be present
        assert ".env" in config.read_file.deny_patterns

    def test_parse_deny_override(self):
        """Should replace deny patterns when using deny_override."""
        config = _parse_config(
            {"read_file": {"allowed": ["/tmp"], "deny_override": ["only_this*"]}}
        )
        assert config.read_file.deny_patterns == ["only_this*"]
        assert ".env" not in config.read_file.deny_patterns


class TestLoadConfig:
    """Test full config loading."""

    def test_load_config_creates_singleton(self, tmp_path):
        """Should create singleton config."""
        reset_config()

        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[read_file]
allowed = ["/tmp"]
""")

        with patch.dict(os.environ, {"GEMINI_MCP_CONFIG": str(config_file)}):
            config1 = load_config()
            assert len(config1.read_file.allowed_paths) == 1

    def test_load_invalid_toml_returns_defaults(self, tmp_path):
        """Should return defaults when TOML is invalid."""
        reset_config()

        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid toml [[[")

        with patch.dict(os.environ, {"GEMINI_MCP_CONFIG": str(config_file)}):
            config = load_config()
            # Should return defaults, not crash
            assert config.read_file.enabled is True


class TestPathTraversalPrevention:
    """Test that path traversal attacks are prevented."""

    def test_dot_dot_traversal_blocked(self, tmp_path):
        """Should block .. traversal attempts."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "data.txt"
        outside_file.write_text("outside content")

        config = ReadFileConfig(allowed_paths=[allowed_dir])

        # Try to access outside file via path traversal
        traversal_path = allowed_dir / ".." / "outside" / "data.txt"
        allowed, msg = config.is_path_allowed(traversal_path)
        assert not allowed
        assert "not under allowed directories" in msg

    def test_absolute_path_outside_blocked(self, tmp_path):
        """Should block absolute paths outside allowed directories."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        config = ReadFileConfig(allowed_paths=[allowed_dir])

        # Try to access /etc/passwd
        allowed, _ = config.is_path_allowed(Path("/etc/passwd"))
        # Should fail because it's not under allowed_dir (may also fail on "File not found" which is fine)
        assert not allowed

    def test_tilde_expansion_validated(self, tmp_path):
        """Should properly expand and validate ~ paths."""
        config = ReadFileConfig(allowed_paths=[tmp_path])

        # ~/ expands to home directory, which is not under tmp_path
        home_file = Path("~/.bashrc")
        if home_file.expanduser().exists():
            allowed, _ = config.is_path_allowed(home_file)
            assert not allowed


class TestWriteFileConfig:
    """Test WriteFileConfig path validation for writes."""

    def test_no_writable_paths_returns_error(self):
        """Should return error when no writable paths configured."""
        config = WriteFileConfig(writable_paths=[])
        allowed, msg = config.is_write_allowed(Path("/some/file.txt"))
        assert not allowed
        assert "not configured" in msg

    def test_disabled_returns_error(self):
        """Should return error when write_file is disabled."""
        config = WriteFileConfig(enabled=False, writable_paths=[Path("/tmp")])
        allowed, msg = config.is_write_allowed(Path("/tmp/test.txt"))
        assert not allowed
        assert "disabled" in msg

    def test_path_under_writable_dir_succeeds(self, tmp_path):
        """Should allow writing to paths under writable directories."""
        config = WriteFileConfig(writable_paths=[tmp_path])
        # Note: file doesn't need to exist for write validation
        target = tmp_path / "new_file.txt"
        allowed, msg = config.is_write_allowed(target)
        assert allowed
        assert msg == ""

    def test_path_outside_writable_dir_fails(self, tmp_path):
        """Should block writes outside writable directories."""
        writable_dir = tmp_path / "writable"
        writable_dir.mkdir()
        outside_file = tmp_path / "outside.txt"

        config = WriteFileConfig(writable_paths=[writable_dir])
        allowed, msg = config.is_write_allowed(outside_file)
        assert not allowed
        assert "not under writable directories" in msg

    def test_deny_pattern_blocked(self, tmp_path):
        """Should block writes matching deny patterns even in writable dirs."""
        config = WriteFileConfig(writable_paths=[tmp_path])
        env_file = tmp_path / ".env"
        allowed, msg = config.is_write_allowed(env_file)
        assert not allowed
        assert "deny pattern" in msg

    def test_symlink_escape_blocked(self, tmp_path):
        """Should block symlink escape attempts."""
        writable_dir = tmp_path / "writable"
        writable_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        # Create symlink inside writable pointing outside
        escape_link = writable_dir / "escape"
        escape_link.symlink_to(outside_dir)

        config = WriteFileConfig(writable_paths=[writable_dir])
        # Try to write via the symlink (using neutral filename to avoid deny pattern match)
        target = escape_link / "data.txt"
        allowed, msg = config.is_write_allowed(target)
        assert not allowed
        assert "not under writable directories" in msg

    def test_nested_path_allowed(self, tmp_path):
        """Should allow writes to nested directories under writable root."""
        config = WriteFileConfig(writable_paths=[tmp_path])
        nested = tmp_path / "deep" / "nested" / "path" / "file.txt"
        allowed, _ = config.is_write_allowed(nested)
        assert allowed

    def test_max_size_stored(self):
        """Should store max size configuration."""
        config = WriteFileConfig(max_size_bytes=2048)
        assert config.max_size_bytes == 2048

    def test_default_max_size(self):
        """Should use default max size."""
        config = WriteFileConfig()
        assert config.max_size_bytes == MAX_WRITE_SIZE_BYTES


class TestWriteFileConfigParsing:
    """Test parsing write_file configuration from TOML."""

    def test_parse_writable_paths(self, tmp_path):
        """Should parse writable paths from config."""
        data = {"write_file": {"writable": [str(tmp_path), "/tmp/output"]}}
        config = _parse_config(data)
        assert len(config.write_file.writable_paths) == 2

    def test_parse_disabled(self):
        """Should parse enabled=false."""
        data = {"write_file": {"enabled": False, "writable": ["/tmp"]}}
        config = _parse_config(data)
        assert config.write_file.enabled is False

    def test_parse_max_size(self):
        """Should parse max_size override."""
        data = {"write_file": {"writable": ["/tmp"], "max_size": 2097152}}
        config = _parse_config(data)
        assert config.write_file.max_size_bytes == 2097152

    def test_parse_additional_deny_patterns(self):
        """Should extend deny patterns with user-specified ones."""
        data = {"write_file": {"writable": ["/tmp"], "deny": ["*.log", "*.tmp"]}}
        config = _parse_config(data)
        assert "*.log" in config.write_file.deny_patterns
        assert "*.tmp" in config.write_file.deny_patterns
        # Default patterns should still be present
        assert ".env" in config.write_file.deny_patterns

    def test_parse_deny_override(self):
        """Should replace deny patterns with deny_override."""
        data = {
            "write_file": {"writable": ["/tmp"], "deny_override": ["only_this_pattern"]}
        }
        config = _parse_config(data)
        assert config.write_file.deny_patterns == ["only_this_pattern"]
        # Default patterns should NOT be present
        assert ".env" not in config.write_file.deny_patterns
