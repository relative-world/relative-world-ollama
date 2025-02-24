import unittest
from unittest.mock import patch, AsyncMock
from pydantic import BaseModel
import asyncio

from relative_world_ollama.client import (
    ollama_generate,
    fix_json_response,
    PydanticOllamaClient,
    get_ollama_client,
)
from relative_world_ollama.exceptions import UnparsableResponseError
from relative_world_ollama.settings import settings


class TestGenerateResponse(BaseModel):
    response: str


class TestClientFunctions(unittest.IsolatedAsyncioTestCase):

    @patch("relative_world_ollama.client.AsyncOllamaClient")
    async def test_ollama_generate(self, MockAsyncOllamaClient):
        mock_client = MockAsyncOllamaClient.return_value
        mock_client.generate = AsyncMock(
            return_value=TestGenerateResponse(response='{"key": "value"}')
        )

        response = await ollama_generate(
            mock_client, "test_model", "test_prompt", "test_system"
        )
        self.assertEqual(response.response, '{"key": "value"}')

    @patch("relative_world_ollama.client.AsyncOllamaClient")
    async def test_fix_json_response(self, MockAsyncOllamaClient):
        mock_client = MockAsyncOllamaClient.return_value
        mock_client.generate = AsyncMock(
            return_value=TestGenerateResponse(response='{"fixed_key": "fixed_value"}')
        )

        response_model = TestGenerateResponse
        fixed_json = await fix_json_response(
            mock_client, '{"bad_json": "value"}', response_model
        )
        self.assertEqual(fixed_json, {"fixed_key": "fixed_value"})

    @patch("relative_world_ollama.client.AsyncOllamaClient")
    async def test_fix_json_response_error(self, MockAsyncOllamaClient):
        mock_client = MockAsyncOllamaClient.return_value
        mock_client.generate = AsyncMock(
            return_value=TestGenerateResponse(response="bad_json")
        )

        response_model = TestGenerateResponse
        with self.assertRaises(UnparsableResponseError):
            await fix_json_response(mock_client, "bad_json", response_model)

    def test_get_ollama_client(self):
        client = get_ollama_client()
        self.assertIsInstance(client, PydanticOllamaClient)


class TestPydanticOllamaClient(unittest.IsolatedAsyncioTestCase):

    @patch("relative_world_ollama.client.AsyncOllamaClient")
    async def test_generate(self, MockAsyncOllamaClient):
        mock_client = MockAsyncOllamaClient.return_value
        mock_client.generate = AsyncMock(
            return_value=TestGenerateResponse(response='{"response": "value"}')
        )

        client = PydanticOllamaClient(settings.base_url, settings.default_model)
        response_model = TestGenerateResponse
        response, validated_response = await client.generate(
            "test_prompt", "test_system", response_model
        )
        self.assertEqual(validated_response.response, "value")


if __name__ == "__main__":
    unittest.main()
