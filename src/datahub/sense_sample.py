from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SenseSample:
    """Unified record describing a single sense-evaluation instance."""

    sample_id: str
    dataset_id: str
    split: str
    language: str
    text_a: str
    text_b: Optional[str]
    lemma: str
    target_span: Optional[Tuple[int, int]]
    sense_tag: Optional[str]
    same_sense: Optional[int]
