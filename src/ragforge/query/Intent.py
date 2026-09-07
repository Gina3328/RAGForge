from __future__ import annotations

from enum import Enum

class Intent(Enum):
    """Query intent classification, used to route to the right retrieval strategy."""

    FACTUAL = "FACTUAL"          # Factual query: looking for a specific fact, definition, or figure.
    PROCEDURAL = "PROCEDURAL"    # Procedural query: asking how to perform a task, step by step.
    COMPARISON = "COMPARISON"    # Comparison query: comparing similarities/differences between things.
    CHITCHAT = "CHITCHAT"        # Chitchat: casual conversation that doesn't need retrieval.