from pydantic import BaseModel

class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True
