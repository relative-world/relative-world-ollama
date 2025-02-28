from pydantic import BaseModel

from relative_world_ollama.client import PydanticOllamaClient
from relative_world_ollama.settings import settings
from relative_world_ollama.tools import tools_to_schema

# Define a Pydantic model to represent the response from the LLM
class EmotiveResponse(BaseModel):
    thought: str
    statement: str
    action: str
    emoji: str


async def main():
    ollama_client = PydanticOllamaClient(settings.base_url, settings.default_model)

    prompt = "What is the capital of France?"

    _, response = await ollama_client.generate(
        system="You are a snarky but helpful AI assistant.",
        prompt=prompt,
        response_model=EmotiveResponse
    )
    print(response)
    # thought="Oh boy, here's an easy one."
    # statement='The capital of France is Paris.'
    # action='Answering a basic geography question.'
    # emoji='🌍✨'


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
