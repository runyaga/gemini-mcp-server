# Gemini MCP Server

A minimal MCP server for Google Gemini using pydantic-ai.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Setup

```bash
# Clone and install
git clone <repo-url> && cd gemini-mcp-server
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Configuration

Add to `~/.claude/settings.json` (Claude Code) or `~/Library/Application Support/Claude/claude_desktop_config.json` (Claude Desktop on macOS):

```json
{
  "mcpServers": {
    "gemini": {
      "type": "stdio",
      "command": "<ABSOLUTE_PATH_TO_REPO>/.venv/bin/python",
      "args": ["-m", "gemini_mcp_server"],
      "env": {
        "GEMINI_API_KEY": "<your_api_key>"
      }
    }
  }
}
```

Replace `<ABSOLUTE_PATH_TO_REPO>` with the full path to your project (e.g., `/Users/you/dev/gemini-mcp-server`).

## Tools

| Tool | Description | Arguments |
|------|-------------|-----------|
| `ask_gemini` | Query Gemini for research, current info, or alternative perspective | `prompt` (required), `model` (optional) |
| `list_gemini_models` | List available Gemini models | none |

## Development

### Run Tests

```bash
pytest -v                    # unit tests
pytest -v -m integration     # integration tests (requires API key)
```

### Architecture

```
Claude <--MCP stdio--> gemini_mcp_server <--pydantic-ai--> Gemini API
```

### Extending

Add new tools in `gemini_mcp_server/__init__.py`:

```python
from pydantic import BaseModel, Field
from mcp.types import Tool, TextContent

class MyToolInput(BaseModel):
    """Input schema for my_tool."""
    arg1: str = Field(description="Description of arg1")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="What this tool does",
            inputSchema=MyToolInput.model_json_schema(),
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "my_tool":
        inputs = MyToolInput.model_validate(arguments)
        # Your implementation
        return [TextContent(type="text", text="result")]
```
