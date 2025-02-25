import logging
from functools import cached_property, partial
from typing import ClassVar, Type, Annotated

from pydantic import BaseModel, PrivateAttr

from relative_world.entity import Entity, BoundEvent
from relative_world_ollama.client import get_ollama_client
from relative_world_ollama.responses import BasicResponse
from relative_world_ollama.tools import tools_to_schema, ToolDefinition

logger = logging.getLogger(__name__)


class OllamaEntity(Entity):
    model: str | None = None
    event_queue: Annotated[list[BoundEvent], PrivateAttr()] = []
    _context: Annotated[list[int] | None, PrivateAttr()] = None
    response_model: ClassVar[Type[BaseModel]] = BasicResponse

    @cached_property
    def ollama_client(self):
        return get_ollama_client()

    def get_prompt(self):
        raise NotImplementedError

    def get_system_prompt(self):
        return "You are a friendly AI assistant."

    async def generate_response(self, prompt, system, response_model):
        return await self.ollama_client.generate(
            prompt=prompt,
            system=system,
            response_model=response_model,
            context=self._context,
        )


    async def update(self):
        rendered_prompt = self.get_prompt()
        system_prompt = self.get_system_prompt()
        logger.debug("Prompt: %s", rendered_prompt)
        logger.debug("System prompt: %s", system_prompt)

        try:
            response_model = self.response_model
        except AttributeError:
            response_model = BasicResponse

        raw_response, response = await self.generate_response(
            prompt=rendered_prompt,
            system=system_prompt,
            response_model=response_model,
        )
        self._context = raw_response.context
        await self.handle_response(response)

        async for event in super().update():
            yield event

    async def handle_response(self, response: BaseModel) -> None:
        pass


class TooledOllamaEntity(OllamaEntity):
    _tools: Annotated[dict[str, ToolDefinition], PrivateAttr()] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tools = {}
        for key, value in self.__class__.__dict__.items():
            if callable(value) and hasattr(value, "_is_tool"):
                tools[key] = getattr(self, key)

        self._tools = tools_to_schema(tools)

    async def generate_response(self, prompt, system, response_model):
        response = await self.ollama_client.generate(
            prompt=prompt,
            system=system,
            response_model=response_model,
            context=self._context,
            tools=self._tools,
        )
        return response