import unittest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from relative_world_ollama.entity import OllamaEntity


class TestResponseModel(BaseModel):
    response: str


class TestOllamaEntity(OllamaEntity):
    response_model = TestResponseModel

    def get_prompt(self):
        return "Test prompt"


class TestOllamaEntityMethods(unittest.TestCase):

    def setUp(self):
        self.entity = TestOllamaEntity(name="Test Entity")

    @patch("relative_world_ollama.entity.get_ollama_client")
    def test_update(self, mock_get_ollama_client):
        mock_client = MagicMock()
        mock_client.generate.return_value = TestResponseModel(response="Test response")
        mock_get_ollama_client.return_value = mock_client

        events = list(self.entity.update())
        self.assertEqual(events, [])

    def test_get_system_prompt(self):
        self.assertEqual(
            self.entity.get_system_prompt(), "You are a friendly AI assistant."
        )

    def test_handle_response(self):
        response = TestResponseModel(response="Test response")
        events = list(self.entity.handle_response(response))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
