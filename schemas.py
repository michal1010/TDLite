from pydantic import BaseModel

class Response(BaseModel):
    content: str