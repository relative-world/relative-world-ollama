import logging
import re
from typing import Type

import orjson
from ollama import AsyncClient as OllamaAsyncClient
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
    except orjson.JSONDecodeError as exc:
        markdown_pattern = "```json(.*)```"
        match = re.search(markdown_pattern, content, re.DOTALL)
        if match:
            try:
                return orjson.loads(match.group(1))
            except orjson.JSONDecodeError as exc2:
                raise exc2 from exc
        raise exc


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


async def fix_json_response(
        client: OllamaAsyncClient, bad_json: str, response_model: Type[BaseModel]
) -> dict:
    """
    Attempt to fix a malformed JSON response using the Ollama client.

    Args:
        client (ollama.AsyncClient): The Ollama client instance.
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
        inline_json_schema_defs(response_model.model_json_schema())
    ).decode("utf-8")
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
