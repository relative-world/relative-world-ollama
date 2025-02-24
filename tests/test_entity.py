import unittest
from unittest.mock import patch, AsyncMock
from pydantic import BaseModel

from relative_world_ollama.entity import OllamaEntity
from relative_world_ollama.responses import BasicResponse


class TestResponseModel(BaseModel):
    response: str


class TestOllamaEntity(OllamaEntity):
    response_model = TestResponseModel

    def get_prompt(self):
        return "Test prompt"


class TestOllamaEntityMethods(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.entity = TestOllamaEntity(name="Test Entity")

    @patch("relative_world_ollama.entity.get_ollama_client")
    async def test_update(self, mock_get_ollama_client):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            return_value=(AsyncMock(context=[1, 2, 3]), TestResponseModel(response="Test response"))
        )
        mock_get_ollama_client.return_value = mock_client

        events = [event async for event in self.entity.update()]
        self.assertEqual(events, [])
        self.assertEqual(self.entity._context, [1, 2, 3])

    async def test_get_system_prompt(self):
        self.assertEqual(
            self.entity.get_system_prompt(), "You are a friendly AI assistant."
        )

    async def test_handle_response(self):
        response = TestResponseModel(response="Test response")
        await self.entity.handle_response(response)
        events = []
        self.assertEqual(events, [])

    async def test_cached_property_ollama_client(self):
        with patch("relative_world_ollama.entity.get_ollama_client") as mock_get_ollama_client:
            mock_client = AsyncMock()
            mock_get_ollama_client.return_value = mock_client
            client = self.entity.ollama_client
            self.assertEqual(client, mock_client)
            mock_get_ollama_client.assert_called_once()

    async def test_update_with_basic_response(self):
        TestOllamaEntity.response_model = BasicResponse
        with patch("relative_world_ollama.entity.get_ollama_client") as mock_get_ollama_client:
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(
                return_value=(AsyncMock(context=[1, 2, 3]), BasicResponse(text="Basic response"))
            )
            mock_get_ollama_client.return_value = mock_client

            events = [event async for event in self.entity.update()]
            self.assertEqual(events, [])
            self.assertEqual(self.entity._context, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()