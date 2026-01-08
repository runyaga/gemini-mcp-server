# Gemini MCP Server

[![CI](https://github.com/runyaga/gemini-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/runyaga/gemini-mcp-server/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/runyaga/gemini-mcp-server/graph/badge.svg)](https://codecov.io/gh/runyaga/gemini-mcp-server)

MCP server for Google Gemini with async deep research capabilities.

## Setup

```bash
git clone https://github.com/runyaga/gemini-mcp-server.git && cd gemini-mcp-server
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Configuration

Add to Claude settings (`~/.claude/settings.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gemini": {
      "type": "stdio",
      "command": "<PATH_TO_REPO>/.venv/bin/python",
      "args": ["-m", "gemini_mcp_server"],
      "env": { "GEMINI_API_KEY": "<your_key>" }
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `ask_gemini` | Query Gemini |
| `list_gemini_models` | List available models |
| `run_code` | Execute Python in Gemini sandbox |
| `start_research` | Start async deep research job |
| `get_research` | Get research job status/result |
| `list_research` | List research jobs |
| `generate_image` | Generate images using Nano Banana |
| `read_file` | Read and analyze local files (requires configuration) |

## Image Generation (Nano Banana)

Generate images using Gemini's Nano Banana capability:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prompt` | Image description | (required) |
| `model` | `gemini-2.5-flash-image` (fast) or `gemini-3-pro-image-preview` (quality) | `gemini-2.5-flash-image` |
| `aspect_ratio` | `1:1`, `16:9`, `9:16`, `4:3`, `3:4` | `1:1` |
| `output_dir` | Directory to save image | `/tmp` |

Images are saved locally and the file path is returned.

## File Analysis (read_file)

The `read_file` tool allows Gemini to analyze local files. For security, this tool requires explicit configuration of allowed directories.

### Security Configuration

Create `~/.config/gemini-mcp-server/config.toml`:

```toml
[read_file]
# Directories where file reading is allowed
allowed = [
    "~/dev",
    "~/Documents/projects",
    "/tmp"
]

# Additional patterns to block (extends defaults)
# deny = ["*custom_blocked*"]

# Or replace default deny patterns entirely
# deny_override = ["only_this*"]
```

### Default Deny Patterns

The following patterns are always blocked, even within allowed directories:

- `.env`, `.env.*`, `.envrc` - Environment files
- `*credentials*`, `*secret*`, `*password*` - Credential files
- `*.pem`, `*.key` - Key files
- `.ssh/*`, `.gnupg/*` - SSH and GPG directories
- `.aws/*`, `.azure/*`, `.gcloud/*` - Cloud credentials
- `.npmrc`, `.pypirc`, `.netrc`, `.git-credentials` - Package manager tokens

### Configuration Options

| Option | Description |
|--------|-------------|
| `enabled` | Enable/disable read_file tool (default: `true`) |
| `allowed` | List of directories where file reading is permitted |
| `deny` | Additional patterns to block (extends defaults) |
| `deny_override` | Replace default deny patterns entirely |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_MCP_CONFIG` | Path to config file (overrides default locations) |

## Usage

```bash
# CLI
export GEMINI_API_KEY="your-key"
python research_cli.py start "topic" --depth thorough
python research_cli.py status <job-id>
python research_cli.py list
```

## Development

```bash
pytest -v                    # unit tests
pytest -v -m integration     # integration tests (requires API key)
```
