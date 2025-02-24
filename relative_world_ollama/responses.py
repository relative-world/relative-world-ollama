from pydantic import BaseModel


class BasicResponse(BaseModel):
    text: str
