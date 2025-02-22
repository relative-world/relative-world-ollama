import logging
from functools import cached_property
from typing import ClassVar, Type, Annotated

import orjson as json
from pydantic import BaseModel, PrivateAttr

from relative_world.entity import Entity, BoundEvent
from relative_world_ollama.client import get_ollama_client

logger = logging.getLogger(__name__)


class BasicResponse(BaseModel):
    text: str


class OllamaEntity(Entity):
    model: str | None = None
    event_queue: Annotated[list[BoundEvent], PrivateAttr()] = []
    response_model: ClassVar[Type[BaseModel]] = BasicResponse

    @cached_property
    def ollama_client(self):
        return get_ollama_client()

    def get_prompt(self):
        raise NotImplementedError

    def get_system_prompt(self):
        return "You are a friendly AI assistant."

    def update(self):
        rendered_prompt = self.get_prompt()
        system_prompt = self.get_system_prompt()
        logger.debug("Prompt: %s", rendered_prompt)
        logger.debug("System prompt: %s", system_prompt)

        try:
            response_model = self.response_model
        except AttributeError:
            response_model = BasicResponse

        response = self.ollama_client.generate(
            prompt=rendered_prompt,
            system=system_prompt,
            response_model=response_model
        )
        if response and (event_iterator := self.handle_response(response)):
            yield from event_iterator

        yield from super().update()

    def handle_response(self, response: BaseModel):
        yield from ()
