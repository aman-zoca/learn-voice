"""JSON story schema validation using Pydantic."""

from pydantic import BaseModel, field_validator
from typing import Literal


VALID_VOICES = ("man", "woman", "kid", "grandma", "grandpa")
VALID_AGES = ("child", "young", "adult", "elderly")
VALID_MOODS = ("calm", "angry", "happy", "sad", "excited", "scared", "whispering")
VALID_LANGUAGES = (
    "english_us", "english_british", "english_indian", "english_australian", "english_irish",
    "hindi", "kannada", "tamil", "telugu", "bengali", "marathi",
)


class Scene(BaseModel):
    speaker: str
    voice: Literal["man", "woman", "kid", "grandma", "grandpa"]
    age: Literal["child", "young", "adult", "elderly"]
    mood: Literal["calm", "angry", "happy", "sad", "excited", "scared", "whispering"]
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Scene text cannot be empty")
        return v


class Story(BaseModel):
    title: str
    language: Literal[
        "english_us", "english_british", "english_indian", "english_australian", "english_irish",
        "hindi", "kannada", "tamil", "telugu", "bengali", "marathi",
    ]
    scenes: list[Scene]

    @field_validator("scenes")
    @classmethod
    def scenes_not_empty(cls, v: list[Scene]) -> list[Scene]:
        if not v:
            raise ValueError("Story must have at least one scene")
        return v
