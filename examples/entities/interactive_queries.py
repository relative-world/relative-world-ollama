import asyncio

from relative_world.entity import Entity
from relative_world.event import Event
from relative_world.world import RelativeWorld
from relative_world_ollama.entity import OllamaEntity
from relative_world_ollama.responses import BasicResponse


class QueryEvent(Event):
    type: str = "QUERY"
    query: str


class QueryActor(Entity):
    name: str = "You"

    async def update(self):
        # Emit a QueryEvent based on user input.
        user_input = input("Enter your query (or press Enter to skip): ")
        if user_input.strip():
            self.emit_event(QueryEvent(query=user_input.strip()))
        # Yield any output from the base class update (if any)
        async for event in super().update():
            yield event

    async def handle_event(self, entity, event):
        # When a ResponseEvent is handled, print it.
        print(f"{self.name}: {event.query}")
        return await super().handle_event(entity, event)


class QueryResponder(OllamaEntity):
    name: str = "Query Responder"

    # response_model is inherited from OllamaEntity; it may be replaced if a different schema is needed

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store the latest query to use as a prompt.
        self._current_query = None

    async def handle_event(self, entity, event):
        # Accept a QueryEvent which contains the user query.
        if isinstance(event, QueryEvent):
            self._current_query = event.query
        return await super().handle_event(entity, event)

    def get_prompt(self):
        # Use the current query if available; otherwise, supply a default prompt.
        if self._current_query:
            prompt = self._current_query
            self._current_query = None  # reset after consumption
            return prompt
        return "<No query>"

    async def handle_response(self, response: BasicResponse):
        # Print the response from the Ollama API.
        print(f"{self.name}: {response.text}")


async def main():
    # Create actor and responder for handling interactive queries.
    query_actor = QueryActor()
    query_responder = QueryResponder()

    # Instantiate the world with the interactive entities.
    world = RelativeWorld(children=[query_actor, query_responder])

    print("Interactive query simulation started. Press Ctrl+C to exit.")
    try:
        # Run a continuous update loop.
        while True:
            await world.step()
    except KeyboardInterrupt:
        print("\nExiting simulation.")


if __name__ == "__main__":
    asyncio.run(main())
