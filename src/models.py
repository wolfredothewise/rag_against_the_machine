"""Pydantic models used across the retrieval and answer pipeline."""

import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Reference to a single source span within a file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without an answer yet."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with the retrieved evidence and final answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset containing both answered and unanswered questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """A search result for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search result enriched with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Collection of ranked search results for a batch of questions."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Answerable search results payload with associated answers."""

    search_results: List[MinimalAnswer]
    k: int
