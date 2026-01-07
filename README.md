# Gemini MCP Server

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
