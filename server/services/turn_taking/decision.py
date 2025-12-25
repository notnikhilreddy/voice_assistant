from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TurnState(str, Enum):
    USER_TALKING = "USER_TALKING"
    USER_THINKING = "USER_THINKING"
    USER_FINISHED = "USER_FINISHED"


@dataclass
class TurnDecision:
    state: TurnState
    p_speech: float
    silence_ms: float
    reason: str = ""


