from ollama import Client

from relative_world_ollama.client import fix_json_response
from pydantic import BaseModel
from relative_world_ollama.exceptions import UnparsableResponseError
from relative_world_ollama.settings import settings


class ExampleResponse(BaseModel):
    data: str

client = Client(host=settings.base_url)
bad_json = '{"data": "Hello, world"'

try:
    fixed_json = fix_json_response(client, bad_json, ExampleResponse)
    print(fixed_json)
except UnparsableResponseError as e:
    print(f"Failed to parse JSON: {e}")
