import json
import logging
from typing import Type, Iterator

from ollama import Client as OllamaClient, GenerateResponse
from pydantic import BaseModel

from relative_world_ollama.exceptions import UnparsableResponseError
from relative_world_ollama.settings import settings

logger = logging.getLogger(__name__)

FIX_JSON_SYSTEM_PROMPT = """
You are a friendly AI assistant.  Your task is to fix poorly formatted json.
Please ensure the user input matches the expected json format and output the corrected structure

The structured output format should match this json schema:

```
{response_model_json_schema}
```
"""


def get_ollama_client():
    return PydanticOllamaClient(
        base_url=settings.base_url, default_model=settings.default_model
    )


def ollama_generate(client: OllamaClient, model: str, prompt: str, system: str) -> GenerateResponse | Iterator[
    GenerateResponse]:
    """
    Generates a response from the Ollama client.

    Args:
        client (OllamaClient): The Ollama client instance.
        model (str): The model to use for generation.
        prompt (str): The prompt to generate a response for.
        system (str): The system context for the generation.

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


def fix_json_response(client: OllamaClient, bad_json: str, response_model: Type[BaseModel]):
    """
    Attempts to fix a malformed JSON response using the Ollama client.

    Args:
        client (OllamaClient): The Ollama client instance.
        bad_json (str): The malformed JSON string.
        response_model (Type[BaseModel]): The Pydantic model to validate the fixed JSON against.

    Returns:
        dict: The fixed JSON data.

    Raises:
        UnparsableResponseError: If the JSON cannot be parsed.
    """
    logger.debug("fix_json::input", extra={"bad_json": bad_json, "response_model": response_model.__name__})

    response_model_json_schema = json.dumps(response_model.model_json_schema(), indent=2)
    system_prompt = FIX_JSON_SYSTEM_PROMPT.format(response_model_json_schema=response_model_json_schema)

    response = client.generate(
        model=settings.json_fix_model,
        prompt=bad_json,
        system=system_prompt,
        keep_alive=settings.ollama_json_model_keep_alive,
    )
    try:
        return json.loads(response.response)
    except json.JSONDecodeError as exc:
        raise UnparsableResponseError(bad_json) from exc


class PydanticOllamaClient:
    """
    A client for interacting with the Ollama API using Pydantic models.
    """

    def __init__(self, base_url, default_model):
        """
        Initializes the PydanticOllamaClient.

        Args:
            base_url (str): The base URL for the Ollama API.
            default_model (str): The default model to use for generation.
        """
        self._client = OllamaClient(host=base_url)
        self.default_model = default_model

    def generate(self, prompt, system, response_model: Type[BaseModel], model: str | None = None):
        """
        Generates a response from the Ollama API and validates it against a Pydantic model.

        Args:
            prompt (str): The prompt to generate a response for.
            system (str): The system context for the generation.
            response_model (Type[BaseModel]): The Pydantic model to validate the response against.
            model (str, optional): The model to use for generation. Defaults to None.

        Returns:
            BaseModel: The validated response model.
        """
        response = ollama_generate(self._client, model or self.default_model, prompt, system)
        response_text = response.response

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = fix_json_response(self._client, response_text, response_model)

        return response_model.model_validate(data)
