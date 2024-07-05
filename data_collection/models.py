from typing import Literal

from pydantic import BaseModel


class Round(BaseModel):
    question: str
    answer: Literal["yes", "no"] | None = None
    guess: str | None = None


class Game(BaseModel):
    keyword: str | None = None
    rounds: list[Round] = []
    win: bool | None = None
