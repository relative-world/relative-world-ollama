import logging
from typing import Type, AsyncIterator, re

import orjson
from ollama import AsyncClient as AsyncOllamaClient, GenerateResponse
from pydantic import BaseModel

from relative_world_ollama.exceptions import UnparsableResponseError
from relative_world_ollama.settings import settings

logger = logging.getLogger(__name__)

FIX_JSON_SYSTEM_PROMPT = """
You are a friendly AI assistant. Your task is to fix poorly formatted json.
Please ensure the user input matches the expected json format and output the corrected structure.
If the input does not match the structure, attempt to re-structure it to match the expected format, 
if that can be done without adding information.

Only respond with json content, any text outside of the structure will break the system.
The structured output format should match this json schema:

{response_model_json_schema}
"""

def maybe_parse_json(content):
    try:
        return orjson.loads(content)
    except orjson.JSONDecodeError:
        markdown_pattern = '```json\n(.*)\n```'
        match = re.search(markdown_pattern, content, re.DOTALL)
        if match:
            return orjson.loads(match.group(1))


def inline_json_schema_defs(schema):
    """Recursively replace $ref references with their definitions from $defs."""
    defs = schema.pop("$defs", {})

    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_key = obj["$ref"].split("/")[-1]
                return resolve_refs(defs.get(ref_key, {}))
            return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    return resolve_refs(schema)


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
        client: AsyncOllamaClient, model: str, prompt: str, system: str
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
        keep_alive=settings.model_keep_alive,
    )
    logger.debug("ollama_generate::output", extra={"response": response})
    return response


async def fix_json_response(
        client: AsyncOllamaClient, bad_json: str, response_model: Type[BaseModel]
) -> dict:
    """
    Attempt to fix a malformed JSON response using the Ollama client.

    Args:
        client (AsyncOllamaClient): The Ollama client instance.
        bad_json (str): The malformed JSON string.
        response_model (Type[BaseModel]): The Pydantic model against which the JSON is validated.

    Returns:
        dict: The corrected JSON structure.

    Raises:
        UnparsableResponseError: If the JSON cannot be parsed even after fixing.
    """
    logger.debug(
        "fix_json_response::input",
        extra={"bad_json": bad_json, "response_model": response_model.__name__},
    )
    response_model_json_schema = orjson.dumps(
        inline_json_schema_defs(response_model.model_json_schema())).decode('utf-8')
    system_prompt = FIX_JSON_SYSTEM_PROMPT.format(
        response_model_json_schema=response_model_json_schema
    )

    response = await client.generate(
        model=settings.json_fix_model,
        prompt=bad_json,
        system=system_prompt,
        keep_alive=settings.model_keep_alive,
    )
    try:
        return maybe_parse_json(response.response)
    except orjson.JSONDecodeError as exc:
        raise UnparsableResponseError(bad_json) from exc


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
            response_model: Type[BaseModel],
            model: str | None = None,
    ) -> BaseModel:
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
        output_schema = orjson.dumps(inline_json_schema_defs(response_model.model_json_schema())).decode('utf-8')
        system_message = (f"{system}\n\nOnly respond with json content, any text outside of the structure will break the system. "
                          f"The structured output format should match this json schema:\n{output_schema}.")

        response = await ollama_generate(
            self._client, model or self.default_model, prompt, system_message
        )
        response_text = response.response

        try:
            data = maybe_parse_json(response_text)
        except orjson.JSONDecodeError:
            data = await fix_json_response(self._client, response_text, response_model)

        return response_model.model_validate(data)
