from relative_world_ollama.client import PydanticOllamaClient
from relative_world_ollama.settings import settings
from relative_world_ollama.tools import tools_to_schema


def check_weather(city: str) -> str:
    return f"The weather in {city} is 76F and sunny."


def check_traffic() -> str:
    return f"accident on highway causing 20 minute delay.  new expected arrival is 40 minutes."


tools = tools_to_schema({
    "check_weather": check_weather,
    "check_traffic": check_traffic
})

ollama_client = PydanticOllamaClient(settings.base_url, settings.default_model)


async def ask_question(question):
    _, response = await ollama_client.generate(
        system="You're a helpful AI assistant",
        prompt=question,
        tools=tools,
    )
    return response


async def main():
    questions = [
        "What's the weather in New York?",
        # The weather in New York is 76F and sunny.

        "How long will it take to get to work?",
        # There is an accident on the highway causing a 20 minute delay. The new expected arrival time to work is
        # 40 minutes.

        "What's the capital of France?",
        # The capital of France is Paris.

        "What's the weather in New York and how long will it take me to get to work? Also what's the capital of France?",
        # The weather in New York is 76F and sunny. There's an accident on the highway which is causing a 20 minute
        # delay, so your new expected arrival time to work is 40 minutes. The capital of France is Paris.
    ]
    for question in questions:
        print(question)
        response = await ask_question(question)
        print(response.text)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
