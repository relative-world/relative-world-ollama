import inspect
import logging
import traceback
from typing import Any, Annotated

from pydantic import BaseModel, PrivateAttr

logger = logging.getLogger(__name__)

TOOL_CALLING_SYSTEM_PROMPT = """
[TOOLS]{tool_definitions_json}[/TOOLS]

To call a tool use the following format:

```
{{
    "tool_calls": [
        {{
            "function_name": <function name>,
            "function_args: <object with function arguments>
        }},
        {{
            "function_name": <function name>,
            "function_args: <object with function arguments>
        }}
    ]
}}
```
Previous tool invocations will be rendered in the TOOL_INVOCATIONS section of the system prompt.
[TOOL_INVOCATIONS]{previous_tool_invocations}[/TOOL_INVOCATIONS]
"""


class ToolCallRequest(BaseModel):
    function_name: str
    function_args: dict[str, Any]


class ToolCallRequestContainer(BaseModel):
    tool_calls: list[ToolCallRequest]


class ToolCallResponse(BaseModel):
    tool_call: ToolCallRequest
    result: str


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
        if parameter.default == inspect.Parameter.empty:
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
        if not hasattr(function, "_tool_schema"):
            function._tool_schema = function_to_schema(function)
        result[name] = function._tool_schema
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
