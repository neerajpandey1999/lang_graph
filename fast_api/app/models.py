from pydantic import BaseModel
from typing import List, Dict

class UserInput(BaseModel):
    user_input: str

class State(BaseModel):
    messages: List[Dict[str, str]]  # Expect a list of dictionaries with 'role' and 'content'
