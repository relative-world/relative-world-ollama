import unittest
from unittest.mock import patch
from pydantic import BaseModel

from relative_world_ollama.client import ollama_generate, fix_json_response, PydanticOllamaClient
from relative_world_ollama.exceptions import UnparsableResponseError
from relative_world_ollama.settings import settings


class TestGenerateResponse(BaseModel):
    response: str

class TestClientFunctions(unittest.TestCase):

    @patch('relative_world_ollama.client.OllamaClient')
    def test_ollama_generate(self, MockOllamaClient):
        mock_client = MockOllamaClient.return_value
        mock_client.generate.return_value = TestGenerateResponse(response='{"key": "value"}')

        response = ollama_generate(mock_client, 'test_model', 'test_prompt', 'test_system')
        self.assertEqual(response.response, '{"key": "value"}')

    @patch('relative_world_ollama.client.OllamaClient')
    def test_fix_json_response(self, MockOllamaClient):
        mock_client = MockOllamaClient.return_value
        mock_client.generate.return_value = TestGenerateResponse(response='{"fixed_key": "fixed_value"}')

        response_model = TestGenerateResponse
        fixed_json = fix_json_response(mock_client, '{"bad_json": "value"}', response_model)
        self.assertEqual(fixed_json, {"fixed_key": "fixed_value"})

    @patch('relative_world_ollama.client.OllamaClient')
    def test_fix_json_response_error(self, MockOllamaClient):
        mock_client = MockOllamaClient.return_value
        mock_client.generate.return_value = TestGenerateResponse(response='bad_json')

        response_model = TestGenerateResponse
        with self.assertRaises(UnparsableResponseError):
            fix_json_response(mock_client, 'bad_json', response_model)

class TestPydanticOllamaClient(unittest.TestCase):

    @patch('relative_world_ollama.client.OllamaClient')
    def test_generate(self, MockOllamaClient):
        mock_client = MockOllamaClient.return_value
        mock_client.generate.return_value = TestGenerateResponse(response='{"response": "value"}')

        client = PydanticOllamaClient(settings.base_url, settings.default_model)
        response_model = TestGenerateResponse
        response = client.generate('test_prompt', 'test_system', response_model)
        self.assertEqual(response.response, 'value')

if __name__ == '__main__':
    unittest.main()
