from pydantic import BaseModel

from relative_world_ollama.entity import TooledOllamaEntity
from relative_world_ollama.tools import tool


class Assistant(TooledOllamaEntity):
    prompt_queue: list[str] = []

    def get_prompt(self):
        if self.prompt_queue:
            return self.prompt_queue.pop(0)
        return "<No user input>"

    def get_system_prompt(self):
        return "You are a friendly AI assistant."

    @tool
    def check_weather(self, city: str) -> str:
        """Check the weather for a given city."""
        return f"The weather in {city} is 76F and sunny."

    @tool
    def check_traffic(self) -> str:
        """Check the current traffic conditions."""
        return "Accident on highway causing 20 minute delay. New expected arrival is 40 minutes."

    def receive_prompt(self, prompt: str):
        print(prompt)
        self.prompt_queue.append(prompt)

    async def handle_response(self, response: BaseModel) -> None:
        print(response.text)


async def main():
    assistant = Assistant()
    questions = [
        "What's the weather in New York?",
        "How long will it take to get to work?",
        "What's the capital of France?",
        "What's the weather in New York and how long will it take me to get to work? Also what's the capital of France?",
    ]
    for question in questions:
        assistant.receive_prompt(question)
        try:
            _, response = await anext(assistant.update())
        except StopAsyncIteration:
            pass


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
