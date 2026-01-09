# Gemini MCP Server

[![CI](https://github.com/runyaga/gemini-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/runyaga/gemini-mcp-server/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/runyaga/gemini-mcp-server/graph/badge.svg)](https://codecov.io/gh/runyaga/gemini-mcp-server)

MCP server for Google Gemini with async deep research, code execution, image generation, and secure file access.

## Installation

```bash
git clone https://github.com/runyaga/gemini-mcp-server.git && cd gemini-mcp-server
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Configuration

### 1. API Key

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Client Setup

Add to `~/.claude/settings.json`:

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

### 3. File Access (Optional)

Create `~/.config/gemini-mcp-server/config.toml`:

```toml
[read_file]
allowed = ["~/dev", "~/Documents"]

[write_file]
writable = ["~/output", "/tmp/gemini-output"]
```

**Security:** Sensitive files (`.env`, `*credentials*`, `*.key`, `.ssh/*`) are blocked by default.

## Tools

| Tool | Description |
|------|-------------|
| `ask_gemini` | Query Gemini models |
| `list_gemini_models` | List available models |
| `run_code` | Execute Python in sandbox (numpy, pandas available) |
| `start_research` | Async deep research with Google Search grounding |
| `get_research` | Get research job status/result |
| `list_research` | List research jobs |
| `generate_image` | Generate images (Nano Banana) |
| `read_file` | Analyze a local file |
| `read_files` | Batch analyze files (max 20, 5MB total) |
| `write_file` | Write to local file |

### Image Generation

- **model**: `gemini-2.5-flash-image` (default) or `gemini-3-pro-image-preview`
- **aspect_ratio**: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`

### Research CLI

```bash
export GEMINI_API_KEY="your-key"
python research_cli.py start "topic" --depth thorough
python research_cli.py status <job-id>
```

## Development

```bash
pytest -v                    # unit tests
pytest -v -m integration     # integration tests (requires API key)
```
