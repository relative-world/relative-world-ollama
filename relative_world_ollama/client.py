import json
import logging
from typing import Type, Iterator

from ollama import Client as OllamaClient, GenerateResponse
from pydantic import BaseModel

from relative_world_ollama.exceptions import UnparsableResponseError
from relative_world_ollama.settings import settings

logger = logging.getLogger(__name__)

FIX_JSON_SYSTEM_PROMPT = """
You are a friendly AI assistant. Your task is to fix poorly formatted json.
Please ensure the user input matches the expected json format and output the corrected structure.

The structured output format should match this json schema:

```
{response_model_json_schema}
```
"""


def get_ollama_client() -> 'PydanticOllamaClient':
    """
    Create and return an instance of PydanticOllamaClient based on settings.

    Returns:
        PydanticOllamaClient: A client configured with the base_url and default_model.
    """
    return PydanticOllamaClient(
        base_url=settings.base_url, default_model=settings.default_model
    )


def ollama_generate(
        client: OllamaClient,
        model: str,
        prompt: str,
        system: str
) -> GenerateResponse | Iterator[GenerateResponse]:
    """
    Generate a response from the Ollama client.

    Args:
        client (OllamaClient): The Ollama client instance.
        model (str): The model name to use.
        prompt (str): The prompt for generation.
        system (str): The system context for generation.

    Returns:
        GenerateResponse | Iterator[GenerateResponse]: The generated response.
    """
    logger.debug("ollama_generate::input", extra={"model": model, "prompt": prompt, "system": system})
    response = client.generate(
        model=model,
        prompt=prompt,
        system=system,
        keep_alive=settings.model_keep_alive,
    )
    logger.debug("ollama_generate::output", extra={"response": response})
    return response


def fix_json_response(
        client: OllamaClient,
        bad_json: str,
        response_model: Type[BaseModel]
) -> dict:
    """
    Attempt to fix a malformed JSON response using the Ollama client.

    Args:
        client (OllamaClient): The Ollama client instance.
        bad_json (str): The malformed JSON string.
        response_model (Type[BaseModel]): The Pydantic model against which the JSON is validated.

    Returns:
        dict: The corrected JSON structure.

    Raises:
        UnparsableResponseError: If the JSON cannot be parsed even after fixing.
    """
    logger.debug("fix_json_response::input", extra={"bad_json": bad_json, "response_model": response_model.__name__})
    response_model_json_schema = json.dumps(response_model.model_json_schema(), indent=2)
    system_prompt = FIX_JSON_SYSTEM_PROMPT.format(response_model_json_schema=response_model_json_schema)

    response = client.generate(
        model=settings.json_fix_model,
        prompt=bad_json,
        system=system_prompt,
        keep_alive=settings.model_keep_alive,
    )
    try:
        return json.loads(response.response)
    except json.JSONDecodeError as exc:
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
        self._client = OllamaClient(host=base_url)
        self.default_model = default_model

    def generate(
            self,
            prompt: str,
            system: str,
            response_model: Type[BaseModel],
            model: str | None = None
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
        output_schema = json.dumps(response_model.model_json_schema())
        system_message = (
                system +
                "\n\nThe structured output format should match this json schema:\n\n```\n" +
                output_schema +
                "\n```"
        )

        response = ollama_generate(
            self._client,
            model or self.default_model,
            prompt,
            system_message
        )
        response_text = response.response

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = fix_json_response(self._client, response_text, response_model)

        return response_model.model_validate(data)
