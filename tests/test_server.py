"""Tests for Gemini MCP Server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set up test environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")


class TestPydanticModels:
    """Test Pydantic input models."""

    def test_ask_gemini_input_validation(self):
        """Should validate ask_gemini input."""
        from gemini_mcp_server import AskGeminiInput

        # Valid input
        inputs = AskGeminiInput(prompt="Hello")
        assert inputs.prompt == "Hello"
        assert inputs.model is None

        # With model
        inputs = AskGeminiInput(prompt="Hello", model="gemini-2.5-pro")
        assert inputs.model == "gemini-2.5-pro"

    def test_ask_gemini_input_requires_prompt(self):
        """Should require prompt field."""
        from pydantic import ValidationError

        from gemini_mcp_server import AskGeminiInput

        with pytest.raises(ValidationError):
            AskGeminiInput()

    def test_ask_gemini_schema_generation(self):
        """Should generate valid JSON schema."""
        from gemini_mcp_server import AskGeminiInput

        schema = AskGeminiInput.model_json_schema()
        assert schema["type"] == "object"
        assert "prompt" in schema["properties"]
        assert "model" in schema["properties"]
        assert "prompt" in schema["required"]

    def test_generate_image_input_validation(self):
        """Should validate generate_image input."""
        from gemini_mcp_server import GenerateImageInput

        # Valid input with defaults
        inputs = GenerateImageInput(prompt="A cat")
        assert inputs.prompt == "A cat"
        assert inputs.model is None
        assert inputs.aspect_ratio == "1:1"
        assert inputs.output_dir is None

        # With all parameters
        inputs = GenerateImageInput(
            prompt="A dog",
            model="gemini-3-pro-image-preview",
            aspect_ratio="16:9",
            output_dir="/custom/path",
        )
        assert inputs.model == "gemini-3-pro-image-preview"
        assert inputs.aspect_ratio == "16:9"
        assert inputs.output_dir == "/custom/path"

    def test_generate_image_input_requires_prompt(self):
        """Should require prompt field."""
        from pydantic import ValidationError

        from gemini_mcp_server import GenerateImageInput

        with pytest.raises(ValidationError):
            GenerateImageInput()

    def test_generate_image_schema_generation(self):
        """Should generate valid JSON schema."""
        from gemini_mcp_server import GenerateImageInput

        schema = GenerateImageInput.model_json_schema()
        assert schema["type"] == "object"
        assert "prompt" in schema["properties"]
        assert "model" in schema["properties"]
        assert "aspect_ratio" in schema["properties"]
        assert "output_dir" in schema["properties"]
        assert "prompt" in schema["required"]


class TestListTools:
    """Test tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_expected_tools(self, mock_env):
        """Should return all available tools."""
        with patch("gemini_mcp_server.GoogleModel"):
            from gemini_mcp_server import list_tools

            tools = await list_tools()

            assert len(tools) == 8
            tool_names = [t.name for t in tools]
            assert "ask_gemini" in tool_names
            assert "list_gemini_models" in tool_names
            assert "run_code" in tool_names
            assert "start_research" in tool_names
            assert "get_research" in tool_names
            assert "list_research" in tool_names
            assert "generate_image" in tool_names
            assert "read_file" in tool_names

            ask_gemini = next(t for t in tools if t.name == "ask_gemini")
            assert "prompt" in ask_gemini.inputSchema["properties"]
            assert "model" in ask_gemini.inputSchema["properties"]

            run_code = next(t for t in tools if t.name == "run_code")
            assert "code" in run_code.inputSchema["properties"]
            assert "context" in run_code.inputSchema["properties"]

            start_research = next(t for t in tools if t.name == "start_research")
            assert "topic" in start_research.inputSchema["properties"]
            assert "depth" in start_research.inputSchema["properties"]

            generate_image = next(t for t in tools if t.name == "generate_image")
            assert "prompt" in generate_image.inputSchema["properties"]
            assert "model" in generate_image.inputSchema["properties"]
            assert "aspect_ratio" in generate_image.inputSchema["properties"]


class TestCallTool:
    """Test tool calling."""

    @pytest.mark.asyncio
    async def test_ask_gemini_calls_agent(self, mock_env):
        """Should call agent.run with prompt and deps."""
        mock_result = MagicMock()
        mock_result.output = "Test response from Gemini"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.create_agent", return_value=mock_agent),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool("ask_gemini", {"prompt": "What is 2+2?"})

            mock_agent.run.assert_called_once_with("What is 2+2?", deps=mock_deps)
            assert result[0].text == "Test response from Gemini"

    @pytest.mark.asyncio
    async def test_ask_gemini_with_model_parameter(self, mock_env):
        """Should use specified model."""
        mock_result = MagicMock()
        mock_result.output = "Response from specific model"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch(
                "gemini_mcp_server.create_agent", return_value=mock_agent
            ) as mock_create,
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool(
                "ask_gemini", {"prompt": "Hello", "model": "gemini-2.5-pro"}
            )

            mock_create.assert_called_once_with("gemini-2.5-pro")
            assert result[0].text == "Response from specific model"

    @pytest.mark.asyncio
    async def test_list_gemini_models(self, mock_env):
        """Should list available models."""
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.0-flash"
        mock_model.display_name = "Gemini 2.0 Flash"
        mock_model.supported_actions = ["generateContent"]

        mock_client = MagicMock()
        mock_client.models.list.return_value = [mock_model]

        mock_deps = MagicMock()
        mock_deps.client = mock_client

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool("list_gemini_models", {})

            assert "gemini-2.0-flash" in result[0].text
            assert "Gemini 2.0 Flash" in result[0].text

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self, mock_env):
        """Should raise error for unknown tool."""
        with patch("gemini_mcp_server.GoogleModel"):
            from gemini_mcp_server import call_tool

            with pytest.raises(ValueError, match="Unknown tool"):
                await call_tool("unknown_tool", {})


class TestGeminiDeps:
    """Test GeminiDeps dataclass."""

    def test_from_env_with_gemini_key(self, mock_env):
        """Should create deps from GEMINI_API_KEY."""
        with patch("gemini_mcp_server.deps.genai.Client") as mock_client:
            from gemini_mcp_server.deps import GeminiDeps

            deps = GeminiDeps.from_env()

            mock_client.assert_called_once_with(api_key="test-api-key")
            assert deps.default_model == "gemini-2.0-flash"
            assert deps.max_retries == 3

    def test_from_env_missing_key(self):
        """Should raise error when no API key."""
        # Use patch.dict to temporarily clear environment variables
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""},
            clear=False,
        ):
            # Need to reload the module to pick up the patched environment
            import importlib

            import gemini_mcp_server.deps as deps_module

            importlib.reload(deps_module)

            with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
                deps_module.GeminiDeps.from_env()


class TestStreaming:
    """Test streaming helper functions."""

    @pytest.mark.asyncio
    async def test_run_with_optional_streaming_no_progress_token(self):
        """Should fall back to regular run when no progress token."""
        mock_result = MagicMock()
        mock_result.output = "Regular response"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_deps = MagicMock()

        from gemini_mcp_server.streaming import run_with_optional_streaming

        result = await run_with_optional_streaming(
            agent=mock_agent,
            prompt="Hello",
            deps=mock_deps,
            progress_token=None,
            session=None,
        )

        mock_agent.run.assert_called_once_with("Hello", deps=mock_deps)
        assert result == "Regular response"


class TestCodeExecutionParser:
    """Test code execution response parsing."""

    def test_parse_successful_execution(self):
        """Should parse successful code execution."""
        from gemini_mcp_server.client import parse_code_execution_response

        # Mock response with executable_code and code_execution_result
        mock_part = MagicMock()
        mock_part.executable_code = MagicMock(code="print('hello')")
        mock_part.code_execution_result = MagicMock(
            outcome="OUTCOME_OK",
            output="hello\n",
        )

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = parse_code_execution_response(mock_response)

        assert result.success is True
        assert result.output == "hello\n"
        assert result.code_executed == "print('hello')"
        assert result.error is None

    def test_parse_failed_execution(self):
        """Should parse failed code execution."""
        from gemini_mcp_server.client import parse_code_execution_response

        mock_part = MagicMock()
        mock_part.executable_code = MagicMock(code="1/0")
        mock_part.code_execution_result = MagicMock(
            outcome="OUTCOME_FAILED",
            output="ZeroDivisionError: division by zero",
        )

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = parse_code_execution_response(mock_response)

        assert result.success is False
        assert result.error == "ZeroDivisionError: division by zero"
        assert result.output is None

    def test_parse_no_candidates(self):
        """Should handle response with no candidates."""
        from gemini_mcp_server.client import parse_code_execution_response

        mock_response = MagicMock()
        mock_response.candidates = []

        result = parse_code_execution_response(mock_response)

        assert result.success is False
        assert "No response candidates" in result.error

    def test_parse_no_content_parts(self):
        """Should handle response with no content parts."""
        from gemini_mcp_server.client import parse_code_execution_response

        mock_candidate = MagicMock()
        mock_candidate.content.parts = []

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = parse_code_execution_response(mock_response)

        assert result.success is False
        assert "No content parts" in result.error

    def test_parse_text_response_no_code_execution(self):
        """Should handle text response when code wasn't executed."""
        from gemini_mcp_server.client import parse_code_execution_response

        mock_part = MagicMock(spec=["text"])
        mock_part.text = "I cannot execute that code."
        mock_part.executable_code = None
        mock_part.code_execution_result = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        result = parse_code_execution_response(mock_response)

        assert result.success is False
        assert "No code executed" in result.error
        assert "I cannot execute that code" in result.error


class TestExecuteCode:
    """Test execute_code function."""

    @pytest.mark.asyncio
    async def test_execute_code_success(self):
        """Should execute code and return result."""
        from gemini_mcp_server.client import execute_code
        from gemini_mcp_server.models import CodeExecutionResult

        # Mock the response
        mock_part = MagicMock()
        mock_part.executable_code = MagicMock(code="print(42)")
        mock_part.code_execution_result = MagicMock(
            outcome="OUTCOME_OK",
            output="42\n",
        )

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await execute_code(
            client=mock_client,
            code="print(42)",
            model="gemini-2.0-flash",
        )

        assert isinstance(result, CodeExecutionResult)
        assert result.success is True
        assert result.output == "42\n"

    @pytest.mark.asyncio
    async def test_execute_code_with_context(self):
        """Should include context in prompt."""
        from gemini_mcp_server.client import execute_code

        mock_part = MagicMock()
        mock_part.executable_code = MagicMock(code="print(42)")
        mock_part.code_execution_result = MagicMock(
            outcome="OUTCOME_OK",
            output="42\n",
        )

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await execute_code(
            client=mock_client,
            code="print(42)",
            context="Calculate the answer",
        )

        # Verify the prompt includes context
        call_args = mock_client.aio.models.generate_content.call_args
        prompt = call_args.kwargs["contents"]
        assert "Calculate the answer" in prompt

    @pytest.mark.asyncio
    async def test_execute_code_api_error(self):
        """Should handle API errors gracefully."""
        from gemini_mcp_server.client import execute_code

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API connection failed")
        )

        result = await execute_code(
            client=mock_client,
            code="print(42)",
        )

        assert result.success is False
        assert "API connection failed" in result.error


class TestRunCodeTool:
    """Test run_code tool handler."""

    @pytest.mark.asyncio
    async def test_run_code_success(self, mock_env):
        """Should execute code and return formatted success response."""
        from gemini_mcp_server.models import CodeExecutionResult

        mock_exec_result = CodeExecutionResult(
            success=True,
            output="42\n",
            code_executed="print(42)",
        )

        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
            patch(
                "gemini_mcp_server.execute_code",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool("run_code", {"code": "print(42)"})

            assert "Code executed successfully" in result[0].text
            assert "42" in result[0].text

    @pytest.mark.asyncio
    async def test_run_code_failure(self, mock_env):
        """Should return formatted error response on failure."""
        from gemini_mcp_server.models import CodeExecutionResult

        mock_exec_result = CodeExecutionResult(
            success=False,
            error="SyntaxError: invalid syntax",
        )

        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
            patch(
                "gemini_mcp_server.execute_code",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool("run_code", {"code": "print(42"})

            assert "Execution failed" in result[0].text
            assert "SyntaxError" in result[0].text

    @pytest.mark.asyncio
    async def test_run_code_with_context(self, mock_env):
        """Should pass context to execute_code."""
        from gemini_mcp_server.models import CodeExecutionResult

        mock_exec_result = CodeExecutionResult(
            success=True,
            output="5050\n",
            code_executed="print(sum(range(101)))",
        )

        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
            patch(
                "gemini_mcp_server.execute_code",
                new_callable=AsyncMock,
                return_value=mock_exec_result,
            ) as mock_execute,
        ):
            from gemini_mcp_server import call_tool

            await call_tool(
                "run_code",
                {"code": "print(sum(range(101)))", "context": "Sum numbers 1-100"},
            )

            mock_execute.assert_called_once()
            call_kwargs = mock_execute.call_args.kwargs
            assert call_kwargs["context"] == "Sum numbers 1-100"


class TestGenerateImage:
    """Test generate_image function."""

    @pytest.mark.asyncio
    async def test_generate_image_success(self):
        """Should generate image and save to file."""
        from gemini_mcp_server.client import generate_image
        from gemini_mcp_server.models import ImageGenerationResult

        # Mock inline_data with image content
        mock_inline_data = MagicMock()
        mock_inline_data.data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data
        mock_part.as_image.return_value = MagicMock()

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await generate_image(
            client=mock_client,
            prompt="A cute cat",
            model="gemini-2.5-flash-image",
        )

        assert isinstance(result, ImageGenerationResult)
        assert result.success is True
        assert result.file_path is not None
        assert result.file_path.startswith("/tmp/gemini_img_")
        assert result.model_used == "gemini-2.5-flash-image"

    @pytest.mark.asyncio
    async def test_generate_image_no_candidates(self):
        """Should handle response with no candidates."""
        from gemini_mcp_server.client import generate_image

        mock_response = MagicMock()
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await generate_image(
            client=mock_client,
            prompt="A cat",
        )

        assert result.success is False
        assert "No response candidates" in result.error

    @pytest.mark.asyncio
    async def test_generate_image_no_image_data(self):
        """Should handle response with no image data."""
        from gemini_mcp_server.client import generate_image

        mock_part = MagicMock()
        mock_part.inline_data = None
        mock_part.text = "I cannot generate that image."

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await generate_image(
            client=mock_client,
            prompt="An inappropriate image",
        )

        assert result.success is False
        assert "No image generated" in result.error
        assert "I cannot generate that image" in result.error

    @pytest.mark.asyncio
    async def test_generate_image_api_error(self):
        """Should handle API errors gracefully."""
        from gemini_mcp_server.client import generate_image

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API connection failed")
        )

        result = await generate_image(
            client=mock_client,
            prompt="A cat",
        )

        assert result.success is False
        assert "API connection failed" in result.error

    @pytest.mark.asyncio
    async def test_generate_image_custom_output_path(self):
        """Should use custom output path when provided."""
        from gemini_mcp_server.client import generate_image

        mock_inline_data = MagicMock()
        mock_inline_data.data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data
        mock_part.as_image.return_value = MagicMock()

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await generate_image(
            client=mock_client,
            prompt="A cat",
            output_path="/tmp/custom_image.png",
        )

        assert result.success is True
        assert result.file_path == "/tmp/custom_image.png"


class TestGenerateImageTool:
    """Test generate_image tool handler."""

    @pytest.mark.asyncio
    async def test_generate_image_tool_success(self, mock_env):
        """Should return formatted success response."""
        from gemini_mcp_server.models import ImageGenerationResult

        mock_result = ImageGenerationResult(
            success=True,
            file_path="/tmp/gemini_img_test123.png",
            model_used="gemini-2.5-flash-image",
        )

        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
            patch(
                "gemini_mcp_server.generate_image",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool("generate_image", {"prompt": "A beautiful sunset"})

            assert "Image generated successfully" in result[0].text
            assert "/tmp/gemini_img_test123.png" in result[0].text
            assert "gemini-2.5-flash-image" in result[0].text

    @pytest.mark.asyncio
    async def test_generate_image_tool_failure(self, mock_env):
        """Should return formatted error response on failure."""
        from gemini_mcp_server.models import ImageGenerationResult

        mock_result = ImageGenerationResult(
            success=False,
            error="Content policy violation",
            model_used="gemini-2.5-flash-image",
        )

        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
            patch(
                "gemini_mcp_server.generate_image",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool(
                "generate_image", {"prompt": "Something inappropriate"}
            )

            assert "Image generation failed" in result[0].text
            assert "Content policy violation" in result[0].text

    @pytest.mark.asyncio
    async def test_generate_image_tool_with_parameters(self, mock_env):
        """Should pass all parameters to generate_image function."""
        from gemini_mcp_server.models import ImageGenerationResult

        mock_result = ImageGenerationResult(
            success=True,
            file_path="/custom/dir/gemini_img_abc123.png",
            model_used="gemini-3-pro-image-preview",
        )

        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
            patch(
                "gemini_mcp_server.generate_image",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_generate,
        ):
            from gemini_mcp_server import call_tool

            await call_tool(
                "generate_image",
                {
                    "prompt": "A landscape",
                    "model": "gemini-3-pro-image-preview",
                    "aspect_ratio": "16:9",
                    "output_dir": "/custom/dir",
                },
            )

            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args.kwargs
            assert call_kwargs["prompt"] == "A landscape"
            assert call_kwargs["model"] == "gemini-3-pro-image-preview"
            assert call_kwargs["aspect_ratio"] == "16:9"
            assert "/custom/dir/" in call_kwargs["output_path"]


class TestReadFileInput:
    """Test ReadFileInput model."""

    def test_read_file_input_validation(self):
        """Should validate read_file input."""
        from gemini_mcp_server.models import ReadFileInput

        # Valid input with defaults
        inputs = ReadFileInput(file_path="/path/to/file.py")
        assert inputs.file_path == "/path/to/file.py"
        assert inputs.prompt == "Analyze this file and summarize its contents."
        assert inputs.model is None

        # With custom prompt
        inputs = ReadFileInput(
            file_path="/path/to/file.py",
            prompt="Find bugs in this code",
            model="gemini-2.5-pro",
        )
        assert inputs.prompt == "Find bugs in this code"
        assert inputs.model == "gemini-2.5-pro"

    def test_read_file_input_requires_file_path(self):
        """Should require file_path field."""
        from pydantic import ValidationError

        from gemini_mcp_server.models import ReadFileInput

        with pytest.raises(ValidationError):
            ReadFileInput()

    def test_read_file_schema_generation(self):
        """Should generate valid JSON schema."""
        from gemini_mcp_server.models import ReadFileInput

        schema = ReadFileInput.model_json_schema()
        assert schema["type"] == "object"
        assert "file_path" in schema["properties"]
        assert "prompt" in schema["properties"]
        assert "model" in schema["properties"]
        assert "file_path" in schema["required"]


class TestReadFileTool:
    """Test read_file tool handler."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, mock_env, tmp_path):
        """Should read file and return Gemini analysis."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'")

        mock_result = MagicMock()
        mock_result.output = "This is a simple function that returns 'world'."
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.create_agent", return_value=mock_agent),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
        ):
            from gemini_mcp_server import call_tool

            result = await call_tool(
                "read_file",
                {"file_path": str(test_file), "prompt": "Explain this code"},
            )

            assert "test.py" in result[0].text
            assert "simple function" in result[0].text
            mock_agent.run.assert_called_once()
            # Verify file contents were passed to agent
            call_args = mock_agent.run.call_args
            assert "def hello():" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, mock_env):
        """Should return error for non-existent file."""
        with patch("gemini_mcp_server.GoogleModel"):
            from gemini_mcp_server import call_tool

            result = await call_tool(
                "read_file",
                {"file_path": "/nonexistent/path/file.txt"},
            )

            assert "Error: File not found" in result[0].text

    @pytest.mark.asyncio
    async def test_read_file_is_directory(self, mock_env, tmp_path):
        """Should return error when path is a directory."""
        with patch("gemini_mcp_server.GoogleModel"):
            from gemini_mcp_server import call_tool

            result = await call_tool(
                "read_file",
                {"file_path": str(tmp_path)},
            )

            assert "Error: Not a file" in result[0].text

    @pytest.mark.asyncio
    async def test_read_file_with_model_parameter(self, mock_env, tmp_path):
        """Should use specified model."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        mock_result = MagicMock()
        mock_result.output = "A greeting"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch(
                "gemini_mcp_server.create_agent", return_value=mock_agent
            ) as mock_create,
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
        ):
            from gemini_mcp_server import call_tool

            await call_tool(
                "read_file",
                {
                    "file_path": str(test_file),
                    "prompt": "Summarize",
                    "model": "gemini-2.5-pro",
                },
            )

            mock_create.assert_called_once_with("gemini-2.5-pro")

    @pytest.mark.asyncio
    async def test_read_file_default_prompt(self, mock_env, tmp_path):
        """Should use default prompt when not provided."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some content")

        mock_result = MagicMock()
        mock_result.output = "Analysis result"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_deps = MagicMock()

        with (
            patch("gemini_mcp_server.GoogleModel"),
            patch("gemini_mcp_server.create_agent", return_value=mock_agent),
            patch("gemini_mcp_server.GeminiDeps.from_env", return_value=mock_deps),
        ):
            from gemini_mcp_server import call_tool

            await call_tool(
                "read_file",
                {"file_path": str(test_file)},
            )

            call_args = mock_agent.run.call_args
            assert "Analyze this file and summarize" in call_args[0][0]


class TestListToolsIncludesReadFile:
    """Test that read_file is included in list_tools."""

    @pytest.mark.asyncio
    async def test_list_tools_includes_read_file(self, mock_env):
        """Should include read_file in tool list."""
        with patch("gemini_mcp_server.GoogleModel"):
            from gemini_mcp_server import list_tools

            tools = await list_tools()

            tool_names = [t.name for t in tools]
            assert "read_file" in tool_names

            read_file = next(t for t in tools if t.name == "read_file")
            assert "file_path" in read_file.inputSchema["properties"]
            assert "prompt" in read_file.inputSchema["properties"]
            assert "model" in read_file.inputSchema["properties"]


class TestIntegration:
    """Integration tests (require API key)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_gemini_call(self):
        """Test real Gemini API calls."""
        import os

        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")

        from gemini_mcp_server import call_tool

        # Test simple math question with default model
        result = await call_tool(
            "ask_gemini", {"prompt": "What is 2+2? Reply with just the number."}
        )
        assert "4" in result[0].text
        assert result[0].type == "text"

        # Test with specific model
        result = await call_tool(
            "ask_gemini",
            {
                "prompt": "What is 3+3? Reply with just the number.",
                "model": "gemini-2.0-flash-lite",
            },
        )
        assert "6" in result[0].text

        # Test list models
        result = await call_tool("list_gemini_models", {})
        assert result[0].type == "text"
        assert "gemini" in result[0].text.lower()
        assert len(result[0].text) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_image_generation(self):
        """Test real image generation with Gemini."""
        import os
        from pathlib import Path

        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")

        from gemini_mcp_server import call_tool

        result = await call_tool(
            "generate_image",
            {"prompt": "A simple red circle on white background"},
        )

        assert "Image generated successfully" in result[0].text
        # Extract file path from response
        for line in result[0].text.split("\n"):
            if line.startswith("File:"):
                file_path = line.split(":", 1)[1].strip()
                assert Path(file_path).exists()
                # Clean up
                Path(file_path).unlink()
                break

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_read_file(self, tmp_path):
        """Test real file reading with Gemini analysis."""
        import os

        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")

        from gemini_mcp_server import call_tool

        # Create a test Python file
        test_file = tmp_path / "example.py"
        test_file.write_text(
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)\n"
        )

        result = await call_tool(
            "read_file",
            {
                "file_path": str(test_file),
                "prompt": "What does this function do? Reply in one sentence.",
            },
        )

        assert result[0].type == "text"
        assert "example.py" in result[0].text
        # Gemini should recognize it's a Fibonacci function
        response_lower = result[0].text.lower()
        assert any(
            word in response_lower for word in ["fibonacci", "sequence", "recursive"]
        )
