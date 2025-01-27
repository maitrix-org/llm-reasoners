from dataclasses import asdict, dataclass
from typing import TypedDict


@dataclass
class AccuracyScore:
    loose: float
    """Loose accuracy: The mean proportion of reference strings found in the generation."""

    strict: float
    """Strict accuracy: The proportion of questions with a loose accuracy of 1.0."""


@dataclass
class RougeScorePart:
    precision: float
    recall: float
    fscore: float


@dataclass
class RougeScore:
    rouge1: RougeScorePart
    rouge2: RougeScorePart
    rougeL: RougeScorePart


@dataclass
class EvaluationSingleScore:
    question_id: str
    acc: float
    rouge: RougeScore
    bleurt: float
    gpt: int


@dataclass
class EvaluationScore:
    acc: AccuracyScore
    rouge: RougeScore
    bleurt: float
    gpt: float
    raw: list[EvaluationSingleScore]

    def to_dict(self, include_raw: bool = False):
        data = asdict(self)
        if not include_raw:
            data.pop('raw', None)
        return data


class Answer(TypedDict):
    """A dictionary of the form ``{"id": "...", "answer": "..."}``."""

    id: str
    answer: str
