import asyncio
import uuid
from typing import List

from pydantic import BaseModel

from relative_world.actor import Actor
from relative_world.event import Event
from relative_world.location import Location
from relative_world.world import RelativeWorld
from relative_world_ollama.client import PydanticOllamaClient
from relative_world_ollama.settings import settings


class DescribedLocation(Location):
    description: str
    connected_locations: List[str] = []


class MovementEvent(Event):
    type: str = "MOVE"
    actor_id: str
    from_location: str | None
    to_location: str


class MovementDecision(BaseModel):
    should_move: bool
    target_location: str | None
    reason: str


class ConversationEvent(Event):
    type: str = "SAY"
    speaker_id: str
    message: str


class ConversationalActor(Actor):
    _ollama_client: PydanticOllamaClient
    memory: List[str] = []
    identity: str

    def __init__(self, world, ollama_client, name, identity):
        super().__init__(world=world, name=name, identity=identity)
        self._ollama_client = ollama_client

    def get_location_info(self, location_id: uuid.UUID) -> dict:
        location = self.world.get_location(location_id)
        connections = self.world.get_connected_locations(location_id)
        return {
            "id": str(location.id),
            "name": location.name,
            "description": location.description,
            "connected_locations": [str(loc.id) for loc in connections]
        }

    async def decide_movement(self, current_info: DescribedLocation) -> MovementDecision:
        # Get the list of other actors in the current location
        other_actors_here = [
            actor.name for actor in self.location.children if actor is not self
        ]

        prompt = f"""
        You are {self.name}, currently in location: {current_info.name}.
        Description: {current_info.description}
        Connected locations: {current_info.connected_locations}
        Other actors in this location: {other_actors_here}
        Recent conversation log: {self.memory[-10:]}

        Decide if you want to move to a connected location or stay.
        """

        response, decision = await self._ollama_client.generate(
            prompt=prompt,
            system="You are an actor in a world. Make movement decisions based on the current location, recent conversations, and the presence of other actors.",
            response_model=MovementDecision
        )
        return decision

    async def act(self):
        if self.location_id:
            current_info = DescribedLocation(**self.get_location_info(self.location_id))
            decision = await self.decide_movement(current_info)

            if decision.should_move and decision.target_location:
                target_loc = self.world.get_location(uuid.UUID(decision.target_location))
                if target_loc:
                    old_loc = self.world.get_location(self.location_id)
                    old_location = self.location_id
                    self.location = target_loc

                    # Notice other actors upon arrival
                    for actor in self.world.get_location(self.location_id).children:
                        if actor is not self:
                            self.memory.append(f"I see {actor.name} here.")

                    print(f"{self.name} moved from {old_loc.name} to {target_loc.name}")
                    yield MovementEvent(
                        actor_id=str(self.id),
                        from_location=str(old_location) if old_location else None,
                        to_location=str(target_loc.id)
                    )

            # Generate a conversation event using the Ollama client
            prompt = f"{self.name} is in {self.location.name}. Generate a message for them to say."
            raw_response, response = await self._ollama_client.generate(prompt=prompt,
                                                                        system=self.create_system_prompt())
            message = response.text
            yield ConversationEvent(speaker_id=str(self.id), message=message)

            # after speaking, add what we said to our own memory
            self.memory.append(f"{message}")

    async def handle_event(self, entity, event: Event):
        if isinstance(event, ConversationEvent) and self.location_id == entity.location_id:
            self.memory.append(f"{entity.name}: {event.message}")
        if isinstance(event, MovementEvent):
            if event.to_location == str(self.location_id):
                self.memory.append(f"{entity.name} has arrived.")
            if event.from_location == str(self.location_id):
                self.memory.append(f"{entity.name} has left.")

        if self is entity:
            print(f"{self.memory[-1]}")

        await super().handle_event(entity, event)

    def create_system_prompt(self):
        other_actors_here = [
            actor.name for actor in self.location.children if actor is not self
        ]
        return f"""
        You are {self.name}.
        Identity: {self.identity}
        Current conversation log: {self.memory}
        Other actors in this location: {other_actors_here}
        Respond to the current situation and conversation.
        """



async def main():
    world = RelativeWorld()
    client = PydanticOllamaClient(base_url=settings.base_url, default_model=settings.default_model)

    locations = {
        "garden": DescribedLocation(
            id=uuid.uuid4(),
            name="Garden",
            description="A beautiful garden with flowers and trees"
        ),
        "house": DescribedLocation(
            id=uuid.uuid4(),
            name="House",
            description="A cozy house with many rooms"
        ),
        "market": DescribedLocation(
            id=uuid.uuid4(),
            name="Market",
            description="A busy marketplace with various vendors"
        )
    }

    for location in locations.values():
        world.add_location(location)

    world.connect_locations(locations["garden"].id, locations["house"].id)
    world.connect_locations(locations["house"].id, locations["market"].id)

    actors = [
        ConversationalActor(world, client, name="Alice", identity="A friendly gardener"),
        ConversationalActor(world, client, name="Bob", identity="A curious housemate"),
        ConversationalActor(world, client, name="Charlie", identity="A busy merchant"),
    ]

    actors[0].location = locations["house"]
    actors[1].location = locations["house"]
    actors[2].location = locations["market"]

    for _ in range(10):
        await world.step()

    for actor in actors:
        print(f"{actor.name}'s memory: {actor.memory}")


if __name__ == "__main__":
    asyncio.run(main())
