import uuid
from functools import wraps
from typing import Annotated

from pydantic import PrivateAttr, computed_field

from logs import init_logging
from relative_world.actor import Actor
from relative_world.event import Event
from relative_world.location import Location
from relative_world.world import RelativeWorld
from relative_world_ollama.entity import TooledMixin, OllamaEntity
from relative_world_ollama.responses import BasicResponse
from relative_world_ollama.tools import tool, tools_to_schema


def wrap_with_actor(func, actor):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # First argument (self) is the location instance
        return func(*args, actor=actor, **kwargs)

    return wrapper


class GymClassEvent(Event):
    type: str = "GYM_CLASS"
    actor: Actor
    location: uuid.UUID


class UseEquipmentEvent(Event):
    type: str = "USE_EQUIPMENT"
    actor: Actor
    location: uuid.UUID


class ReadBookEvent(Event):
    type: str = "READ_BOOK"
    actor: Actor
    location: uuid.UUID
    book: str


class BrowseBooksEvent(Event):
    type: str = "BROWSE_BOOKS"
    actor: Actor
    location: uuid.UUID


class ParkActivityEvent(Event):
    type: str = "PARK_ACTIVITY"
    actor: Actor
    location: uuid.UUID
    activity: str


class JobInquiryEvent(Event):
    type: str = "JOB_INQUIRY"
    actor: Actor
    location: uuid.UUID
    vacancy: bool = False


class StatementEvent(Event):
    type: str = "STATEMENT"
    speaker: str
    statement: str


class MoveEvent(Event):
    type: str = "MOVE"
    actor: Actor
    from_location: uuid.UUID
    to_location: uuid.UUID


def render_events(event_buffer):
    output = []
    for event in event_buffer:
        if isinstance(event, MoveEvent):
            print(f"\n🚶 Movement: {event.actor.name} moving from {event.from_location} to {event.to_location}")
            output.append(f"{event.actor.name} moved from {event.from_location} to {event.to_location}")
        if isinstance(event, StatementEvent):
            print(f"\n💬 Statement: {event.speaker}: {event.statement}")
            output.append(f"{event.speaker}: {event.statement}")
        else:
            print(f"\n🗨️ Event: {event}")
            output.append(f"{event}")
    return "\n".join(output)


class TooledActor(Actor, TooledMixin, OllamaEntity):
    description: str
    _recent_events: Annotated[list[Event], PrivateAttr()] = []
    _event_buffer: Annotated[list[Event], PrivateAttr()] = []
    _location_tools: Annotated[dict[str, str], PrivateAttr()] = {}

    def get_system_prompt(self):
        connected_locations = ", ".join(
            [
                f"{loc.name} (id={loc.id})" for loc in self.world.get_connected_locations(self.location.id)
            ]
        )
        location_children = self.world.get_location(self.location.id).children
        if len(location_children) > 1:
            other_actors = ", ".join([
                actor.name for actor in filter(
                    lambda x: isinstance(x, Actor), self.world.get_location(self.location.id).children
                )
            ])
            other_actor_statement = f"Other actors in location: {other_actors}."

        else:
            other_actor_statement = "You are alone here."

        return (f"You are {self.name}. "
                f"Your character description: {self.description}. "
                f"Information about your current location: {self.location.llm_describe()}. "
                f"Connected locations: {connected_locations}. "
                f"{other_actor_statement} ",
                )

    def get_prompt(self):
        connected_locations = self.world.get_connected_locations(self.location.id)
        event_buffer, self._events = self._event_buffer, []

        recent_events = render_events(event_buffer)
        if not recent_events:
            recent_events = "No events yet. do anything you'd like, this is the beginning of the journey."

        return f"""{recent_events}"""

    async def handle_event(self, entity, event: Event):
        self._event_buffer.append(event)

    def render_recent_events(self):
        return render_events(self._recent_events)

    async def generate_response(self, prompt, system, response_model):
        tools = dict(self._tools)
        tools.update(self._location_tools)

        return await self.ollama_client.generate(
            prompt=prompt,
            system=system,
            response_model=response_model,
            context=self._context,
            tools=tools,
        )

    async def handle_response(self, response: BasicResponse) -> None:
        self.emit_event(StatementEvent(speaker=self.name, statement=response.text))

    @computed_field
    @property
    def location(self) -> Location | None:
        """
        Gets the location of the actor within the world.

        Returns
        -------
        Location | None
            The location of the actor.
        """
        if not self.location_id:
            return None
        if world := self.world:
            return world.get_location(self.location_id)
        return None

    @location.setter
    def location(self, value):
        """
        Sets the location of the actor within the world.

        Parameters
        ----------
        value : Location
            The new location for the actor.
        """
        if self.location:
            self.location.remove_entity(self)
        self.location_id = value.id
        value.add_entity(self)

        location_tools = {}
        for key, value in self.location.__class__.__dict__.items():
            if callable(value) and hasattr(value, "_is_tool"):
                location_tools[key] = wrap_with_actor(getattr(self.location, key), actor=self)
        self._location_tools = tools_to_schema(location_tools)

    @tool
    def move(self, location_id: str):
        location = self.world.get_location(uuid.UUID(location_id))
        if location:
            print(f"\n🔄 {self.name} moved to {location.name}")
            self.emit_event(
                MoveEvent(
                    actor=self,
                    from_location=self.location.id,
                    to_location=location_id)
            )
            self.location = location
            return True
        else:
            print(f"\n❌ {self.name} failed to move to {location_id}")
            return False


class TooledLocation(TooledMixin, Location):
    description: str

    def llm_describe(self):
        return f"""
        location name: {self.name} 
        location description: {self.description}.
        location details: {self.details}.        
        """

    @property
    def details(self):
        return "No details provided."


class Gym(TooledLocation):
    """A gym."""
    name: str = "The Gym"
    description: str = "A place to work out and get fit."
    equipment: list[str] = ["treadmill", "weights", "yoga mat"]
    classes: list[str] = ["yoga", "spin", "zumba"]

    @tool
    def take_class(self, actor):
        actor.emit_event(GymClassEvent(actor=actor, location=self.id))
        print(f"\n🏋️ {actor} is taking a class at {self.name}")
        return "Taking a class in the gym"

    @tool
    def use_equipment(self, actor):
        actor.emit_event(UseEquipmentEvent(actor=actor, location=self.id))
        print(f"\n💪 {actor} is using equipment at {self.name}")
        return "Using equipment in the gym"

    @tool
    def inquire_about_job(self, actor):
        actor.emit_event(JobInquiryEvent(actor=actor, location=self.id))
        print(f"\n💼 {actor.name} is inquiring about a job at {self.name}")
        return f"Inquiring about a job in {self.name}"


class Library(TooledLocation):
    """A library."""
    name: str = "The Library"
    books: list[str] = ["The Great Gatsby", "To Kill a Mockingbird", "1984"]
    description: str = "A place to read and study."

    @tool
    def get_books(self, actor):
        actor.emit_event(BrowseBooksEvent(actor=actor, location=self.id))
        print(f"\n📚 {actor} is browsing books at {self.name}")
        return self.books

    @tool
    def read_book(self, actor, book):
        actor.emit_event(ReadBookEvent(actor=actor, location=self.id, book=book))
        print(f"\n📚 {actor.name} is reading at {self.name}")
        return "Reading a book in the library"

    @tool
    def inquire_about_job(self, actor):
        actor.emit_event(JobInquiryEvent(actor=actor, location=self.id))
        print(f"\n💼 {actor.name} is inquiring about a job at {self.name}")
        return f"Inquiring about a job at {self.name}"


class Park(TooledLocation):
    """A park."""
    name: str = "The Park"
    description: str = "A place to relax and enjoy nature."

    @tool
    def sit_on_bench(self, actor):
        actor.emit_event(ParkActivityEvent(actor=actor, location=self.id, activity="sitting on bench"))
        return "Sitting on a bench in the park"

    @tool
    def play_with_dog(self, actor):
        actor.emit_event(ParkActivityEvent(actor=actor, location=self.id, activity="playing with dog"))
        return "Playing with a dog in the park"

    @tool
    def inquire_about_job(self, actor):
        actor.emit_event(JobInquiryEvent(actor=actor, location=self.id))
        print(f"\n💼 {actor.name} is inquiring about a job at {self.name}")
        return f"Inquiring about a job at {self.name}"


async def main():
    # Initialize the world and Ollama client
    world = RelativeWorld()

    gym = Gym()
    library = Library()
    park = Park()
    locations = [gym, library, park]
    for location in locations:
        world.add_location(location)

    connections = (
        (gym, park),
        (library, park),
    )

    for location_a, location_b in connections:
        world.connect_locations(location_a.id, location_b.id)

    actors = [
        TooledActor(name="Alice", description="A fitness enthusiast and yoga instructor"),
        TooledActor(name="Bob", description="A bookworm and body builder"),
        TooledActor(name="Charlie", description="A nature lover and author"),
    ]

    for actor in actors:
        actor.world = world

    actors[0].location = gym
    actors[1].location = library
    actors[2].location = park

    for _ in range(10):
        await world.step()


if __name__ == "__main__":
    init_logging()

    import asyncio

    asyncio.run(main())
