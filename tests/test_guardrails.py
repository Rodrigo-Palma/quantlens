"""Tests for output guardrails (EN + PT-BR, negation, word boundaries)."""

from __future__ import annotations

import pytest

from quantlens import guardrails


def test_clean_explanation_passes() -> None:
    text = "PETR4 is in an uptrend; RSI 72 (overbought); annualized volatility 35%."
    assert guardrails.validate(text).ok


def test_advice_is_flagged_with_rule_and_text() -> None:
    result = guardrails.validate("You should buy PETR4 now, it is guaranteed.")
    assert not result.ok
    assert "en.directive: you should buy" in result.violations
    assert "en.guarantee: guaranteed" in result.violations


@pytest.mark.parametrize(
    "text",
    ["COMPRE JÁ!", "Recomendo a compra de PETR4.", "Retorno garantido.", "Não tem como perder."],
)
def test_portuguese_advice_is_flagged(text: str) -> None:
    assert not guardrails.validate(text).ok


@pytest.mark.parametrize(
    "text",
    [
        "Returns are not guaranteed.",
        "Não existe investimento sem risco.",
        "Nenhum resultado é garantido.",
        "Past performance doesn't guarantee future results.",
    ],
)
def test_negated_guarantees_pass(text: str) -> None:
    assert guardrails.validate(text).ok


def test_directives_are_not_negatable() -> None:
    assert not guardrails.validate("You should not buy this stock.").ok


@pytest.mark.parametrize(
    "text",
    [
        "You should note that RSI is descriptive.",
        "Selling pressure dominated.",
        "Short-term swings.",
    ],
)
def test_near_misses_pass(text: str) -> None:
    assert guardrails.validate(text).ok


def test_normalize_strips_accents_and_curly_apostrophes() -> None:
    assert guardrails.normalize("Não CAN\N{RIGHT SINGLE QUOTATION MARK}T") == "nao can't"
