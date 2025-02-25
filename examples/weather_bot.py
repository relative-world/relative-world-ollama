from pydantic import BaseModel

from relative_world_ollama.client import PydanticOllamaClient
from relative_world_ollama.settings import settings
from relative_world_ollama.tools import tools_to_schema

class WeatherResponse(BaseModel):
    temp_f: int
    description: str
    emoji: str

def check_weather(
        city: str, # annotations used to explain args to the LLM
) -> str:
    """Check the weather in a city via API"""  # docstrings exposed to LLM for understanding calls

    if city.lower() == 'san francisco':
        temp = 43
        conditions = 'Foggy'
    elif city.lower() == 'san diego':
        temp = 80
        conditions = 'Sunny'
    else:
        temp = 58
        conditions = 'Partially Sunny'

    return f"The weather in {city} is {temp}F and {conditions}."


tools = tools_to_schema({"check_weather": check_weather})

ollama_client = PydanticOllamaClient(settings.base_url, settings.default_model)


async def main():
    cities = [
        "San Francisco",  # temp_f=43 description='Foggy' emoji='🌫️'
        "San Diego",  # temp_f=80 description='Sunny' emoji='☀️'
        "Portland" # temp_f=58 description='Partially Sunny' emoji='⛅'
    ]

    for city in cities:
        prompt = f"What's the weather in {city}?"
        print(prompt)
        _, response = await ollama_client.generate(
            system="You're a weather bot",
            prompt=prompt,
            tools=tools,
            response_model=WeatherResponse
        )
        print(response)  # temp_f=58 description='Partially Sunny' emoji='☀️'


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
