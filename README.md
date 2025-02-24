# Relative World Ollama

Relative World Ollama is a Python simulation extension for interacting with the Ollama API. It leverages structured prompt generation, response validation using Pydantic, and JSON repair functionality for a dynamic simulation framework built upon Relative World.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
  - [Basic Usage of `OllamaEntity`](#basic-usage-of-ollamaentity)
  - [Custom Response Model](#custom-response-model)
  - [Handling JSON Fixing](#handling-json-fixing)
- [Configuration](#configuration)
- [Testing](#testing)
- [License](#license)

## Overview

Relative World Ollama extends the Relative World simulation framework by integrating API-driven entity behavior using the Ollama model. It provides a base `OllamaEntity` class that wraps prompt generation, response validation with Pydantic models, and error handling (including fixing malformed JSON responses).

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install the project, run:

```bash
poetry install
```

## Usage Examples

### Basic Usage of `OllamaEntity`

The following example shows how to subclass `OllamaEntity` and implement the `get_prompt` method for generating a query.

```python
from relative_world_ollama.entity import OllamaEntity

class MyOllamaEntity(OllamaEntity):
    def get_prompt(self):
        return "What is the capital of France?"

    def handle_response(self, response):
        print(response.text)

# Create and update the entity
entity = MyOllamaEntity(name="CapitalQuery")
list(entity.update()) # Use list to iterate over the update generator
```

### Custom Response Model

This example demonstrates how to use a custom Pydantic response model.

```python
from pydantic import BaseModel
from relative_world_ollama.entity import OllamaEntity

class CustomResponse(BaseModel):
    answer: str

class CustomOllamaEntity(OllamaEntity):
    response_model = CustomResponse

    def get_prompt(self):
        return "What is the capital of Germany?"

    def handle_response(self, response: CustomResponse):
        print(f"The answer is: {response.answer}")

entity = CustomOllamaEntity(name="CapitalQuery")
list(entity.update())
```

### Handling JSON Fixing

When the API returns malformed JSON, the `fix_json_response` function can repair it:

```python
from ollama import Client

from relative_world_ollama.json import fix_json_response
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
```

## Configuration

The module is configured in the `relative_world_ollama/settings.py` file. Key settings include:
- `base_url` – Base URL for the Ollama API.
- `default_model` – Default model for generation.
- `json_fix_model` – Model used to fix malformed JSON responses.
- `model_keep_alive` – Duration (in seconds) to keep the model alive during execution.

You can override these settings via environment variables prefixed with `relative_world_ollama_`.

## Testing

Tests are executed using [pytest](https://docs.pytest.org/). Run the tests with:

```bash
poetry run pytest
```

## License

This project is licensed under the MIT License.
