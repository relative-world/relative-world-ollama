import asyncio
import uuid
from typing import List, Annotated

from pydantic import BaseModel, PrivateAttr

from relative_world.actor import Actor
from relative_world.event import Event
from relative_world.location import Location
from relative_world.world import RelativeWorld
from relative_world_ollama.client import PydanticOllamaClient
from relative_world_ollama.entity import OllamaEntity
from relative_world_ollama.settings import settings


class MovementEvent(BaseModel):
    actor_id: str
    from_location: str | None
    to_location: str


class DescribedLocation(Location):
    description: str
    connected_locations: List[str] = []


class MovementDecision(BaseModel):
    should_move: bool
    target_location: str | None
    reason: str


class WanderingActor(OllamaEntity, Actor):

    def get_location_info(self, location_id: uuid.UUID) -> dict:
        """Get information about a specific location."""
        location = self.world.get_location(location_id)
        connections = self.world.get_connected_locations(location_id)
        return {
            "id": str(location.id),
            "name": location.name,
            "description": "A location",
            "connected_locations": [str(loc.id) for loc in connections]
        }

    async def decide_movement(self, current_info: DescribedLocation) -> MovementDecision:
        prompt = f"""
        You are {self.name}, currently in location: {current_info.name}.
        Description: {current_info.description}
        Connected locations: {current_info.connected_locations}

        Decide if you want to move to a connected location or stay.
        """

        response, decision = await self.ollama_client.generate(
            prompt=prompt,
            system="You are an actor in a world. Make movement decisions based on the current location.",
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
                    old_location = self.location_id
                    self.location = target_loc
                    yield MovementEvent(
                        actor_id=str(self.id),
                        from_location=str(old_location) if old_location else None,
                        to_location=str(target_loc.id)
                    )


class LoggingActor(Actor):
    def __init__(self, world: RelativeWorld):
        super().__init__(world=world)

    async def handle_event(self, entity, event: Event):
        print(event)


async def main():
    # Initialize the world and Ollama client
    world = RelativeWorld()
    client = PydanticOllamaClient(base_url=settings.base_url, default_model=settings.default_model)

    # Create locations
    locations = {
        "garden": DescribedLocation(
            private=False,
            id=uuid.uuid4(),
            name="Garden",
            description="A beautiful garden with flowers and trees"
        ),
        "house": DescribedLocation(
            private=False,
            id=uuid.uuid4(),
            name="House",
            description="A cozy house with many rooms"
        ),
        "market": DescribedLocation(
            private=False,
            id=uuid.uuid4(),
            name="Market",
            description="A busy marketplace with various vendors"
        )
    }

    # Add locations to world and create connections
    for location in locations.values():
        world.add_location(location)

    world.connect_locations(locations["garden"].id, locations["house"].id)
    world.connect_locations(locations["house"].id, locations["market"].id)

    # Create actors
    actors = [
        WanderingActor(name="Alice"),
        WanderingActor(name="Bob"),
        WanderingActor(name="Charlie"),
    ]

    log_actor = LoggingActor(world)
    log_actor.location = world

    # Set initial locations
    actors[0].location = locations["garden"]
    actors[1].location = locations["house"]
    actors[2].location = locations["market"]

    for _ in range(10):
        await world.step()


if __name__ == "__main__":
    asyncio.run(main())
