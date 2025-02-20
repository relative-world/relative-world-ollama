import logging
from functools import cached_property
from typing import ClassVar, Type

from pydantic import BaseModel

from relative_world.entity import Entity
from relative_world.event import BoundEvent
from relative_world_ollama.client import PydanticOllamaClient
from relative_world_ollama.settings import settings

logger = logging.getLogger(__name__)


class OllamaEntity(Entity):
    name: str
    model: str | None = None
    event_queue: list[BoundEvent] = []
    response_model: ClassVar[Type[BaseModel]]

    @cached_property
    def ollama_client(self):
        return PydanticOllamaClient(
            base_url=settings.base_url, default_model=settings.default_model
        )

    def get_prompt(self):
        raise NotImplementedError

    def get_system_prompt(self):
        return """You are a friendly AI assistant."""

    def update(self):
        rendered_prompt = self.get_prompt()
        system_prompt = self.get_system_prompt()
        logger.debug("Prompt: %s", rendered_prompt)
        logger.debug("System prompt: %s", system_prompt)

        response = self.ollama_client.generate(
            prompt=rendered_prompt,
            system=system_prompt,
            response_model=self.response_model
        )
        if response:
            yield from self.handle_response(response)

        yield from super().update()

    def handle_response(self, response: BaseModel):
        yield from ()
