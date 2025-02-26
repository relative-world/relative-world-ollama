import uuid

from relative_world.actor import Actor
from relative_world.location import Location
from relative_world.world import RelativeWorld
from relative_world_ollama.entity import TooledOllamaEntity
from relative_world_ollama.tools import tool


class DescribedLocation(Location):
    description: str


class WanderingActor(Actor, TooledOllamaEntity):

    def get_prompt(self):
        connected_locations = self.world.get_connected_locations(self.location.id)

        return f"""
        You are {self.name}, currently in location: {self.location.name}.
        Description: {self.location.description}
                
        Connected locations: {connected_locations}
        
        """

    def get_system_prompt(self):
        return (
            "You are a wandering actor in a world."
            "You can move to a new location or look around."
            "Only invoke one tool and do it once, you will have future chances to perform tool calls later."
            "If you have invoked a tool, do not invoke another, instead provide a response."
            "Providing a response will clear the tool calls and allow you to invoke another tool on the next invocation."
            "Other actors are also wandering around the world."
            "If you hog the tool calls, no one else can act and the world will be dead."
        )

    async def handle_response(self, response):
        print(f"{self.name} says {response.text}")

    @tool
    def move(self, location_id: str) -> str:
        """
        Move to a new place.  valid location_id is a connected location.
        """
        location = self.world.get_location(uuid.UUID(location_id))
        if location:
            print(f"{self.name} is moving from {self.location} to {location}.")
            self.location = location
            return True
        else:
            print(f"{self.name} cannot move to {location}.")
            return False

    @tool
    def look(self) -> str:
        """Look around."""
        print(f"{self.name} is looking around.")
        return "It's a beautiful day. You should go somewhere new"


async def main():
    # Initialize the world and Ollama client
    world = RelativeWorld()

    # Create locations
    locations = {
        "garden": DescribedLocation(
            name="Garden",
            description="A beautiful garden with flowers and trees"
        ),
        "house": DescribedLocation(
            name="House",
            description="A cozy house with many rooms"
        ),
        "market": DescribedLocation(
            name="Market",
            description="A busy marketplace with various vendors"
        )
    }

    # Add locations to world and create connections
    for location in locations.values():
        world.add_location(location)

    world.connect_locations(locations["garden"].id, locations["house"].id)
    world.connect_locations(locations["house"].id, locations["market"].id)

    actors = [
        WanderingActor(name="Alice"),
        WanderingActor(name="Bob"),
        WanderingActor(name="Charlie"),
    ]
    for actor in actors:
        actor.world = world

    actors[0].location = locations["garden"]
    actors[1].location = locations["house"]
    actors[2].location = locations["market"]

    for _ in range(10):
        await world.step()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
