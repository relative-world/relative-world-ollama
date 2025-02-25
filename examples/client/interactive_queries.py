"""An interactive LLM client with no memory across invocations.

Example output:

>>> Whats the capital of france?
thought='This is a basic geography question.'
statement='The capital of France is Paris.'
action='Answer the question.'
emoji='🇫🇷'

>>> what's the height of the empire state building?
thought='The Empire State Building is a well-known landmark, and its height is a common trivia question.'
statement='The Empire State Building stands at 1,454 feet (443 meters) tall.'
action='Provide the exact height of the Empire State Building.'
emoji=':building:'

>>> What's the airspeed velocity of an unladen swallow?
thought='Oh boy, here we go with another classic Monty Python reference.'
statement="The airspeed velocity of an unladen European swallow is approximately 11 meters per second,
           but that's a tricky question without more context!"
action='Provide the canonical answer from Monty Python and also mention that it depends on where you
       get your information.'
emoji='🤣🐦'

"""
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

    while True:
        prompt = input(">>> ")
        if prompt.lower() == "Q":
            break

        _, response = await ollama_client.generate(
            system="You are a snarky but helpful AI assistant.",
            prompt=prompt,
            response_model=EmotiveResponse
        )

        print(f"{response}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
