import logging
from typing import Type, AsyncIterator

import orjson
from ollama import AsyncClient as AsyncOllamaClient, GenerateResponse
from pydantic import BaseModel

from relative_world_ollama.responses import BasicResponse, TooledResponse
from relative_world_ollama.json import maybe_parse_json, inline_json_schema_defs, fix_json_response
from relative_world_ollama.settings import settings
from relative_world_ollama.tools import TOOL_CALLING_SYSTEM_PROMPT, ToolCallRequestContainer, call_tool, \
    ToolCallResponse, ToolDefinition

logger = logging.getLogger(__name__)


def get_ollama_client() -> "PydanticOllamaClient":
    """
    Create and return an instance of PydanticOllamaClient based on settings.

    Returns:
        PydanticOllamaClient: A client configured with the base_url and default_model.
    """
    return PydanticOllamaClient(
        base_url=settings.base_url, default_model=settings.default_model
    )


async def ollama_generate(
        client: AsyncOllamaClient,
        model: str,
        prompt: str,
        system: str,
        context: list[int] | None = None,
) -> GenerateResponse | AsyncIterator[GenerateResponse]:
    """
    Generate a response from the Ollama client.

    Args:
        client (AsyncOllamaClient): The Ollama client instance.
        model (str): The model name to use.
        prompt (str): The prompt for generation.
        system (str): The system context for generation.

    Returns:
        GenerateResponse | AsyncIterator[GenerateResponse]: The generated response.
    """
    logger.debug(
        "ollama_generate::input",
        extra={"model": model, "prompt": prompt, "system": system},
    )
    response = await client.generate(
        model=model,
        prompt=prompt,
        system=system,
        context=context,
        keep_alive=settings.model_keep_alive,
    )
    logger.debug("ollama_generate::output", extra={"response": response})
    return response


class PydanticOllamaClient:
    """
    A client for interacting with the Ollama API using Pydantic model validation.

    This client wraps the Ollama API client and handles generating responses and validating
    them against provided Pydantic models.
    """

    def __init__(self, base_url: str, default_model: str):
        """
        Initialize the PydanticOllamaClient instance.

        Args:
            base_url (str): The base URL for the Ollama API.
            default_model (str): The default model name for generation.
        """
        self._client = AsyncOllamaClient(host=base_url)
        self.default_model = default_model

    async def generate(
            self,
            prompt: str,
            system: str,
            response_model: Type[BaseModel] = BasicResponse,
            model: str | None = None,
            tools: list[ToolDefinition] | None = None,
            previous_tool_invocations: list[ToolCallResponse] | None = None,
            context: list[int] | None = None,
    ) -> tuple[GenerateResponse | AsyncIterator[GenerateResponse], BaseModel]:
        """
        Generate a response from Ollama API and validate it against a Pydantic model.

        The method sends a prompt along with a system message (which is appended with a JSON
        schema for structured output) to the Ollama API. It then attempts to parse and validate
        the output. If the parsing fails, a fix is attempted using fix_json_response.

        Args:
            prompt (str): The prompt for generating the response.
            system (str): The system context for generation.
            response_model (Type[BaseModel]): The Pydantic model for validating the response.
            model (str, optional): The model name to use for generation. Defaults to None.

        Returns:
            BaseModel: The validated response model.
        """
        if tools:
            response_model = TooledResponse[response_model]

        previous_tool_invocations = previous_tool_invocations or []
        output_schema = orjson.dumps(
            inline_json_schema_defs(response_model.model_json_schema())
        ).decode("utf-8")

        system_message = ""
        if tools:
            system_message = TOOL_CALLING_SYSTEM_PROMPT.format(
                tool_definitions_json=orjson.dumps({name: tool.model_dump() for name, tool in tools.items()}).decode("utf-8"),
                previous_tool_invocations=orjson.dumps([tc.model_dump() for tc in previous_tool_invocations]).decode("utf-8"),
            )

        system_message += (
            f"\n{system}\n\nOnly respond with json content, any text outside of the structure will break the system. "
            f"The structured output format should match this json schema:\n{output_schema}."
        )
        response = await ollama_generate(
            client=self._client,
            model=model or self.default_model,
            prompt=prompt,
            system=system_message,
            context=context,
        )
        response_text = response.response

        try:
            data = maybe_parse_json(response_text)
        except orjson.JSONDecodeError:
            data = await fix_json_response(self._client, response_text, response_model)

        response_obj = response_model.model_validate(data)
        if response_obj.tool_call:
            call_tool(tools, response_obj.tool_call)
        return response, response_obj.response
