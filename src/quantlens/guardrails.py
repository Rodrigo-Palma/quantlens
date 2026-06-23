"""Output guardrails for generated explanations.

Keep explanations descriptive: never let the model emit investment advice or
guarantees. The API uses this to reject an LLM output and fall back to the
deterministic explainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Phrases that turn a description into advice or a guarantee.
_DISALLOWED = (
    "should buy",
    "should sell",
    "buy now",
    "sell now",
    "recommend buying",
    "recommend selling",
    "you should",
    "guaranteed",
    "can't lose",
    "cannot lose",
    "risk-free",
    "sure thing",
)


@dataclass
class GuardrailResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def validate(explanation: str) -> GuardrailResult:
    """Flag investment-advice / guarantee phrasing in an explanation."""
    text = explanation.lower()
    violations = [phrase for phrase in _DISALLOWED if phrase in text]
    return GuardrailResult(ok=not violations, violations=violations)
