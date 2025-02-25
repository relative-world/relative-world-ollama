import asyncio
import random
import uuid
from typing import List, Optional, Union, Dict

from pydantic import BaseModel, Field

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

class Item(BaseModel):
    name: str
    description: str
    value: int

class InventoryEntry(BaseModel):
    item: Item
    quantity: int

class InventoryQueryEvent(Event):
    type: str = "INVENTORY_QUERY"
    asker_id: str
    target_id: str

class InventoryResponseEvent(Event):
    type: str = "INVENTORY_RESPONSE"
    responder_id: str
    target_id: str
    inventory: List[InventoryEntry]

class TradeOffer(BaseModel):
    offered_items: List[InventoryEntry]
    requested_items: List[InventoryEntry]
    reason: str

class TradeCounterOffer(Event):
    type: str = "COUNTER_TRADE"
    original_trade_id: str  # To track which trade this is countering
    initiator_id: str
    target_id: str
    offer: TradeOffer

class TradeEvent(Event):
    type: str = "TRADE"
    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    initiator_id: str
    target_id: str
    offer: TradeOffer

class TradeDecision(BaseModel):
    accept_trade: bool
    reason: str

class TradingActor(Actor):
    _ollama_client: PydanticOllamaClient
    memory: List[str] = []
    identity: str
    inventory: List[InventoryEntry] = []
    pending_trades: List[TradeOffer] = []
    known_inventories: Dict[str, List[InventoryEntry]] = {}
    known_items: List[Item] = []

    def __init__(self, world, ollama_client, name, identity, starting_inventory):
        super().__init__(world=world, name=name, identity=identity, inventory=starting_inventory)
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

        conn_locations = [f"{loc.name}(id={loc.id})" for loc in self.location.connected_locations]

        system = f"""
        You are {self.name}, a character in a dynamic world.
        Identity and Background: {self.identity}
        Current location: {current_info.name} - {current_info.description}
        Connected locations: {conn_locations}
        Other actors in this location: {other_actors_here}
        Recent conversation log: {self.memory[-10:]}

        Consider your relationships, motivations, and the current situation.
        Make movement decisions based on the location, recent conversations, and the presence of other actors.
        """

        response, decision = await self._ollama_client.generate(
            prompt=prompt,
            system=system,
            response_model=MovementDecision
        )
        return decision

    async def decide_trade(self, offer: TradeOffer, other_actor: "TradingActor") -> TradeDecision | TradeOffer:
        """Use LLM to decide whether to accept, reject, or counter the trade."""
        prompt = f"""
        {other_actor.name} is offering you: {[f"{e.quantity}x {e.item.name}" for e in offer.offered_items]}
        You would have to give them: {[f"{e.quantity}x {e.item.name}" for e in offer.requested_items]}
        Reason for the trade: {offer.reason}
        Your current inventory: {[f"{e.quantity}x {e.item.name}" for e in self.inventory]}

        You can:
        1. Accept the trade as is
        2. Make a counter-offer
        3. Reject the trade

        What would you like to do?
        """
        system = f"""
        You are {self.name}, a character in a dynamic world.
        Identity and Background: {self.identity}
        Current location: {self.location.name} - {self.location.description}
        Other actors in this location: {[actor.name for actor in self.location.children if actor is not self]}
        Recent conversation log: {self.memory[-10:]}

        Consider your relationships, motivations, and the current situation.
        Make trade decisions based on the offer, your inventory, and the presence of other actors.
        """
        response, decision = await self._ollama_client.generate(
            prompt=prompt,
            system=system,
            response_model=Union[TradeDecision, TradeOffer]
        )
        return decision

    async def act(self):
        # First, query inventories of others in the same location
        other_actors = [actor for actor in self.location.children if actor is not self]
        for other in other_actors:
            if str(other.id) not in self.known_inventories:
                yield InventoryQueryEvent(
                    asker_id=str(self.id),
                    target_id=str(other.id)
                )

        if self.location_id:
            current_info = DescribedLocation(**self.get_location_info(self.location_id))
            decision = await self.decide_movement(current_info)

            if decision.should_move and decision.target_location:
                target_loc = self.world.get_location(decision.target_location)
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


            if other_actors and random.random() < 0.3:  # 30% chance to initiate trade
                target = random.choice(other_actors)
                trade_offer = await self.create_trade_offer(target)
                if trade_offer:
                    yield TradeEvent(
                        initiator_id=str(self.id),
                        target_id=str(target.id),
                        offer=trade_offer
                    )

    async def handle_event(self, entity, event: Event):
        if isinstance(event, InventoryQueryEvent) and event.target_id == str(self.id):
            # Respond to inventory queries
            self.emit_event(InventoryResponseEvent(
                responder_id=str(self.id),
                target_id=event.asker_id,
                inventory=self.inventory
                )
            )

        elif isinstance(event, InventoryResponseEvent) and event.target_id == str(self.id):
            # Store the inventory information
            self.known_inventories[event.responder_id] = event.inventory
            # Learn about new items
            for entry in event.inventory:
                if entry.item not in self.known_items:
                    self.known_items.append(entry.item)

        if isinstance(event, TradeEvent) and event.target_id == str(self.id):
            print(f"\n🤝 {entity.name} proposes trade to {self.name}:")
            print(f"Offering: {[f'{e.quantity}x {e.item.name}' for e in event.offer.offered_items]}")
            print(f"Requesting: {[f'{e.quantity}x {e.item.name}' for e in event.offer.requested_items]}")

            decision = await self.decide_trade(event.offer, entity)

            if isinstance(decision, TradeOffer):
                print(f"💭 {self.name} makes a counter-offer")
                self.emit_event(TradeEvent(
                    initiator_id=str(self.id),
                    target_id=str(entity.id),
                    offer=decision
                ))
                self.memory.append(f"Made counter-offer to {entity.name}")
            elif decision.accept_trade:
                if self.can_complete_trade(event.offer):
                    self.complete_trade(event.offer, entity)
                    print(f"✅ {self.name} accepted trade from {entity.name}")
                    self.memory.append(f"Completed trade with {entity.name}: {event.offer.reason}")
                else:
                    print(f"❌ {self.name} couldn't complete trade (insufficient items)")
            else:
                print(f"❌ {self.name} rejected trade: {decision.reason}")
        if isinstance(event, ConversationEvent) and self.location_id == entity.location_id:
            self.memory.append(f"{entity.name}: {event.message}")
        if isinstance(event, MovementEvent):
            if event.to_location == str(self.location_id):
                self.memory.append(f"{entity.name} has arrived.")
            if event.from_location == str(self.location_id):
                self.memory.append(f"{entity.name} has left.")
        if isinstance(event, TradeEvent) and event.target_id == str(self.id):
            # Decide whether to accept the trade
            trade_decision = await self.decide_trade(event.offer, entity)
            if trade_decision.accept_trade:
                if self.can_complete_trade(event.offer):
                    self.complete_trade(event.offer, entity)
                    self.memory.append(f"Accepted trade from {entity.name}: {event.offer.reason}")
                    print(f"{self.name} accepted trade from {entity.name}: {event.offer.reason}")
                else:
                    self.memory.append(f"Could not complete trade from {entity.name} due to insufficient items.")
                    print(f"Could not complete trade from {entity.name} due to insufficient items.")
            else:
                self.memory.append(f"Rejected trade from {entity.name}: {trade_decision.reason}")
                print(f"{self.name} rejected trade from {entity.name}: {trade_decision.reason}")

        if self is entity and self.memory:
            print(f"({self.location.name}) {self.memory[-1]}")

        await super().handle_event(entity, event)


    def add_item(self, item: Item, quantity: int = 1):
        for entry in self.inventory:
            if entry.item.name == item.name:
                entry.quantity += quantity
                return
        self.inventory.append(InventoryEntry(item=item, quantity=quantity))

    def remove_item(self, item_name: str, quantity: int = 1) -> bool:
        for entry in self.inventory:
            if entry.item.name == item_name:
                if entry.quantity >= quantity:
                    entry.quantity -= quantity
                    if entry.quantity == 0:
                        self.inventory.remove(entry)
                    return True
        return False

    async def create_trade_offer(self, target: "ConversationalActor") -> Optional[TradeOffer]:
        """Create trade offer using knowledge of target's inventory."""
        known_inventory = self.known_inventories.get(str(target.id), [])

        prompt = f"""
        You are {self.name}. You want to trade with {target.name}.
        Your inventory: {[f"{e.quantity}x {e.item.name}" for e in self.inventory]}
        {target.name}'s known inventory: {[f"{e.quantity}x {e.item.name}" for e in known_inventory]}
        Available items in the world: {[item.name for item in self.known_items]}

        Create a trade that:
        1. Only offers items you actually have
        2. Requests items you know {target.name} has or might have
        3. Makes sense for both parties
        """
        system = "You are a trading character. Create fair and realistic trade offers based on known inventories.  Favor even trades and promote fair exchanges."

        response, trade_offer = await self._ollama_client.generate(
            prompt=prompt,
            system=system,
            response_model=TradeOffer
        )

        # Validate the trade offer
        if not self.can_complete_trade(trade_offer):
            return None

        return trade_offer

    async def decide_trade(self, offer: TradeOffer, other_actor: "TradingActor") -> TradeDecision:
        """Use LLM to decide whether to accept a trade."""
        prompt = f"""
        {other_actor.name} is offering you: {[f"{e.quantity}x {e.item.name}" for e in offer.offered_items]}
        You would have to give them: {[f"{e.quantity}x {e.item.name}" for e in offer.requested_items]}
        Reason for the trade: {offer.reason}
        Your current inventory: {[f"{e.quantity}x {e.item.name}" for e in self.inventory]}
        Do you accept this trade?
        """
        system = f"""
        You are a trading character. Evaluate if the trade is beneficial for you.
        Consider what you need and what you are giving away.  Be willing to make unfair trades in your own favor.
        """

        response, trade_decision = await self._ollama_client.generate(
            prompt=prompt,
            system=system,
            response_model=TradeDecision
        )
        return trade_decision

    def can_complete_trade(self, offer: TradeOffer) -> bool:
        """Check if the actor has the items to complete the trade."""
        for requested_item in offer.requested_items:
            found = False
            for inventory_item in self.inventory:
                if inventory_item.item.name == requested_item.item.name:
                    if inventory_item.quantity >= requested_item.quantity:
                        found = True
                        break
            if not found:
                return False
        return True

    def complete_trade(self, offer: TradeOffer, other_actor: "TradingActor"):
        """Complete the trade by exchanging items."""
        # Give items to the other actor
        for requested_item in offer.requested_items:
            self.remove_item(requested_item.item.name, requested_item.quantity)
            other_actor.add_item(requested_item.item, requested_item.quantity)

        # Receive items from the other actor
        for offered_item in offer.offered_items:
            self.add_item(offered_item.item, offered_item.quantity)
            other_actor.remove_item(offered_item.item.name, offered_item.quantity)

    def create_system_prompt(self):
        other_actors_here = [
            actor.name for actor in self.location.children if actor is not self
        ]
        inventory_desc = [f"{entry.quantity}x {entry.item.name}" for entry in self.inventory]
        return f"""
        You are {self.name}.
        Identity and Background: {self.identity}
        Current inventory: {inventory_desc}
        Current conversation log: {self.memory}
        Other actors present: {other_actors_here}
        Location: {self.location.name} - {self.location.description}

        Consider your relationships and motivations when speaking or trading.
        Your speech should reflect your personality and background.
        Respond to the current situation and conversation while staying true to your character.
        """

async def main():
    world = RelativeWorld()
    client = PydanticOllamaClient(base_url=settings.base_url, default_model=settings.default_model)

    items = {
        "apple": Item(name="Apple", description="A fresh red apple", value=1),
        "bread": Item(name="Bread", description="A warm loaf of bread", value=2),
        "flower": Item(name="Flower", description="A beautiful flower", value=1),
        "herbs": Item(name="Herbs", description="Fresh medicinal herbs", value=3),
        "wine": Item(name="Wine", description="A bottle of local wine", value=5),
        "cheese": Item(name="Cheese", description="Aged farm cheese", value=4),
        "fabric": Item(name="Fabric", description="Fine woven fabric", value=6),
        "pottery": Item(name="Pottery", description="Handcrafted ceramic pot", value=4),
        "tools": Item(name="Tools", description="Basic farming tools", value=7),
        "spices": Item(name="Spices", description="Exotic cooking spices", value=8),
        "panda plush": Item(name="Panda Plush", description="A cute panda plushie", value=10),
        "massive diamond": Item(name="Massive Diamond", description="A huge diamond", value=100),
        "Shimmering Brew": Item(name="Shimmering Brew", description="A magical brew that captures starlight", value=15),
        "Magical Artifact": Item(name="Magical Artifact", description="A rare magical artifact", value=50),
        "Moonlight Cloak": Item(name="Moonlight Cloak", description="A cloak woven from moonlight", value=250),
        "Clockwork Machine": Item(name="Clockwork Machine", description="An intricate clockwork machine", value=200),
        "Mystical Trillium": Item(name="Mystical Trillium", description="A rare mystical flower", value=30),
        "Enchanted Stones": Item(name="Enchanted Stones", description="Stones imbued with magic", value=20),
        "Magical Artifact Supplies": Item(name="Magical Artifact Supplies", description="Supplies for creating magical artifacts", value=25),
    }

    locations = {
        "enchanted_grove": DescribedLocation(
            id=uuid.uuid4(),
            name="Enchanted Grove Market",
            description="A mystical marketplace where ancient trees intertwine with merchant stalls, glowing with fairy lights"
        ),
        "artificers_row": DescribedLocation(
            id=uuid.uuid4(),
            name="Artificer's Row",
            description="A bustling street lined with stalls selling magical devices and enchanted trinkets"
        ),
        "mystic_square": DescribedLocation(
            id=uuid.uuid4(),
            name="Mystic Square",
            description="The central marketplace where magical merchants gather beneath floating lanterns"
        ),
        "alchemists_corner": DescribedLocation(
            id=uuid.uuid4(),
            name="Alchemist's Corner",
            description="A fragrant section filled with bubbling potions and exotic ingredients"
        ),
        "ethereal_bazaar": DescribedLocation(
            id=uuid.uuid4(),
            name="Ethereal Bazaar",
            description="A shimmering market space where rare magical artifacts are traded"
        ),
        "spellweavers_circle": DescribedLocation(
            id=uuid.uuid4(),
            name="Spellweaver's Circle",
            description="A circular marketplace where magic users gather to trade enchanted goods"
        ),
    }

    actors = [
        TradingActor(
            world, client,
            name="Eldara",
            identity="""Elven potion master with recipes passed down through generations.
                Known for her shimmering brews that capture starlight.
                Seeking rare ingredients for the Moonfire Festival celebrations.
                Eldara is a friend of Thorna and is known to have a sweet tooth.
                Eldara is looking for the character Mystwoven so she can offer the panda plush for trade.
                Mystwoven is courting Thorna and is looking for a gift for her.
                She needs a gift for Grimm.
                """,
            starting_inventory=[
                InventoryEntry(item=items["herbs"], quantity=80),
                InventoryEntry(item=items["Shimmering Brew"], quantity=50),
                InventoryEntry(item=items["spices"], quantity=20),
                InventoryEntry(item=items["panda plush"], quantity=1)
            ]
        ),
        TradingActor(
            world, client,
            name="Grimm",
            identity="""Dwarven artificer crafting enchanted tools and devices.
                His mechanical marvels are sought after throughout the realm.
                Preparing special festival contraptions for the celebration.
                Grimm loves cheese.
                """,
            starting_inventory=[
                InventoryEntry(item=items["Clockwork Machine"], quantity=500),
                InventoryEntry(item=items["tools"], quantity=100),
                InventoryEntry(item=items["apple"], quantity=10),
                InventoryEntry(item=items["wine"], quantity=30)
            ]
        ),
        TradingActor(
            world, client,
            name="Zephyr",
            identity="""Sylph merchant who trades in exotic magical artifacts.
                Floats gracefully between stalls on currents of wind.
                Collecting mystical trinkets for the festival's grand display.
                Knows Grimm loves Cheese.
                Zephyr is walking around greeting everyone they see. 
            """,
            starting_inventory=[
                InventoryEntry(item=items["Magical Artifact"], quantity=50),
                InventoryEntry(item=items["cheese"], quantity=100),
                InventoryEntry(item=items["fabric"], quantity=400),
                InventoryEntry(item=items["spices"], quantity=51),
                InventoryEntry(item=items["pottery"], quantity=13),
                InventoryEntry(item=items["Enchanted Stones"], quantity=20),
            ]
        ),
        TradingActor(
            world, client,
            name="Thorna",
            identity="""Half-orc herbalist specializing in rare magical plants.
                Her knowledge of mystical flora is unmatched in the market.
                Preparing special bloom displays for the festival grounds.
                Thorna loves pandas.  She is being courted by Mystwoven.""",
            starting_inventory=[
                InventoryEntry(item=items["pottery"], quantity=65),
                InventoryEntry(item=items["wine"], quantity=15),
                InventoryEntry(item=items["Mystical Trillium"], quantity=30),
                InventoryEntry(item=items["bread"], quantity=43)
            ]
        ),
        TradingActor(
            world, client,
            name="Mystwoven",
            identity="""Ethereal weaver who crafts clothing from spun moonlight.
                Creates magical garments that shimmer with enchantment.
                Wants to buy something special to give to Thorna and 
                is seeking her out to find out what she likes.
                Mystwoven is courting Thorna.
                you can find Thorna at enchanted_grove.
                you can find Grimm at artificers_row
                """,
            starting_inventory=[
                InventoryEntry(item=items["Moonlight Cloak"], quantity=15),
                InventoryEntry(item=items["tools"], quantity=43),
                InventoryEntry(item=items["pottery"], quantity=21),
                InventoryEntry(item=items["wine"], quantity=14)
            ]
        ),
        TradingActor(
            world, client,
            name="Rune",
            identity="""Gnomish runesmith who deals in enchanted stones.
                Each of his creations holds unique magical properties.
                Headed into town to find gifts for his loved ones.""",
            starting_inventory=[
                InventoryEntry(item=items["spices"], quantity=30),
                InventoryEntry(item=items["fabric"], quantity=20),
                InventoryEntry(item=items["tools"], quantity=13),
                InventoryEntry(item=items["wine"], quantity=24),
                InventoryEntry(item=items["pottery"], quantity=11),
                InventoryEntry(item=items["massive diamond"], quantity=1),
                InventoryEntry(item=items["Magical Artifact Supplies"], quantity=20),
            ]
        )
    ]

    for name, location in locations.items():
        world.add_location(location)

    actors[0].location = locations["alchemists_corner"]  # Eldara in her corner
    actors[1].location = locations["artificers_row"]  # Grimm in his row
    actors[2].location = locations["mystic_square"]  # Zephyr in the square
    actors[3].location = locations["enchanted_grove"]  # Thorna in the grove
    actors[4].location = locations["alchemists_corner"]  # Mystwoven in the bazaar
    actors[5].location = locations["mystic_square"]  # Rune in the circle

    # Connect all locations logically
    location_connections = [
        ("enchanted_grove", "spellweavers_circle"),
        ("spellweavers_circle", "mystic_square"),
        ("spellweavers_circle", "ethereal_bazaar"),
        ("mystic_square", "alchemists_corner"),
        ("artificers_row", "spellweavers_circle"),
        ("artificers_row", "enchanted_grove"),
        ("ethereal_bazaar", "alchemists_corner")
    ]
    for loc1, loc2 in location_connections:
        world.connect_locations(locations[loc1].id, locations[loc2].id)

    for actor in actors:
        print(f"{actor.name}'s inventory: {[f'{e.quantity}x {e.item.name}' for e in actor.inventory]}")

    for _ in range(5):
        await world.step()

    for actor in actors:
        print(f"{actor.name}'s inventory: {[f'{e.quantity}x {e.item.name}' for e in actor.inventory]}")
        print(f"{actor.name}'s memory: {actor.memory}")


if __name__ == "__main__":
    asyncio.run(main())