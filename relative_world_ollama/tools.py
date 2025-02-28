import inspect
import logging
import traceback
from functools import wraps
from typing import Any, Annotated

from pydantic import BaseModel, PrivateAttr


logger = logging.getLogger(__name__)

TOOL_CALLING_SYSTEM_PROMPT = """
[TOOLS]{tool_definitions_json}[/TOOLS]

To call a tool, include the following in your response:

```
{{
    "tool_call": {{
        "function_name": <function name>,
        "function_args: <object with function arguments>
    }}
}}
```
You will be allowed a single tool call and a single response per iteration.
The output of your tool call will be provided to you in the next iteration.
On the next iteration, any state changes induced by tool calls will be reflected in your world.
If you do not wish to call a tool, simply provide a response.
"""


class ToolCallRequest(BaseModel):
    function_name: str
    function_args: dict[str, Any]


class ToolCallRequestContainer(BaseModel):
    tool_calls: list[ToolCallRequest]


class ToolCallResponse(BaseModel):
    tool_call: ToolCallRequest
    result: Any


class FunctionSchemaFunction(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]


class FunctionSchema(BaseModel):
    type: str = "function"
    _callable: Annotated[callable, PrivateAttr()]
    function: FunctionSchemaFunction


class ParameterDefinition(BaseModel):
    type: str
    properties: dict[str, Any]


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


def py_type_to_param_type(annotation):
    if annotation == str:
        return "string"
    elif annotation == int:
        return "integer"
    elif annotation == float:
        return "number"
    elif annotation == bool:
        return "boolean"
    elif annotation == list:
        return "array"
    elif annotation == dict:
        return "object"
    else:
        return "string"


def function_to_schema(function: callable) -> FunctionSchema:
    """It'll do it's best to convert a function to a schema."""
    signature = inspect.signature(function)
    required = []
    for name, parameter in signature.parameters.items():
        if parameter.default == inspect.Parameter.empty and name not in {'self', 'actor'}:
            required.append(name)

    fs = FunctionSchema(
        function=FunctionSchemaFunction(
            name=function.__name__,
            description=function.__doc__ or "",
            parameters={
                name: {
                    "type": py_type_to_param_type(parameter.annotation),
                    "description": parameter.annotation.__name__,
                }
                for name, parameter in signature.parameters.items()
                if name not in {'self', 'actor'}
            },
            required=required,
        )
    )
    fs._callable = function
    return fs


def tools_to_schema(tools: dict[str, callable]) -> dict[str, FunctionSchema]:
    """It'll do it's best to convert a function to a schema."""
    result = {}
    for name, function in tools.items():
        result[name] = function_to_schema(function)
    return result


def call_tool(tools, tool_call):
    logger.debug(f"Tool call: {tool_call}")
    try:
        function = tools[tool_call.function_name]._callable
        result = function(**(tool_call.function_args))
    except Exception:
        result = traceback.format_exc()
    return ToolCallResponse(
        tool_call=tool_call,
        result=result,
    )


def tool(func):
    """Decorator to mark a function as a tool"""
    func._is_tool = True
    return func


def wrap_with_actor(func, actor):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # First argument (self) is the location instance
        return func(*args, actor=actor, **kwargs)

    return wrapper
