from relative_world_ollama.client import PydanticOllamaClient
from relative_world_ollama.responses import BasicResponse
from relative_world_ollama.settings import settings
from relative_world_ollama.tools import tools_to_schema


def check_weather(city: str) -> str:
    return f"The weather in {city} is 76F and sunny."


tools = tools_to_schema({"check_weather": check_weather})

ollama_client = PydanticOllamaClient(settings.base_url, settings.default_model)


async def main():
    _, response = await ollama_client.generate(
        system="You're a weather bot",
        prompt="What's the weather in New York?",
        tools=tools, response_model=BasicResponse
    )
    print(response.text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
