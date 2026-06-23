"""Offline evaluation harness for explanations.

Runs the deterministic explainer over a small eval set and checks two things:
faithfulness (the text reflects the input signals) and guardrails (no advice).
Exits non-zero when the pass rate is below the threshold; wired into CI so a
regression in the explanation logic fails the build — no network or LLM needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantlens import guardrails
from quantlens.explain import rule_based

THRESHOLD = 1.0  # every case must pass


@dataclass(frozen=True)
class Case:
    ticker: str
    rsi: float
    mom: float
    vol: float


@dataclass
class CaseResult:
    ticker: str
    passed: bool
    reasons: list[str]


EVAL_SET: tuple[Case, ...] = (
    Case("PETR4", rsi=75.0, mom=0.08, vol=0.35),
    Case("VALE3", rsi=25.0, mom=-0.06, vol=0.40),
    Case("ITUB4", rsi=50.0, mom=0.01, vol=0.22),
    Case("BBAS3", rsi=68.0, mom=-0.02, vol=0.28),
)


def _faithfulness_issues(text: str, case: Case) -> list[str]:
    lowered = text.lower()
    issues: list[str] = []
    if "rsi" not in lowered:
        issues.append("missing RSI")
    if "volatility" not in lowered:
        issues.append("missing volatility")
    trend = "uptrend" if case.mom > 0 else "downtrend"
    if trend not in lowered:
        issues.append(f"missing {trend}")
    return issues


def run() -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in EVAL_SET:
        text = rule_based(case.ticker, case.rsi, case.mom, case.vol)
        reasons = _faithfulness_issues(text, case)
        guard = guardrails.validate(text)
        reasons.extend(f"guardrail: {v}" for v in guard.violations)
        results.append(CaseResult(case.ticker, not reasons, reasons))
    return results


def main() -> None:
    results = run()
    passed = sum(result.passed for result in results)
    total = len(results)
    rate = passed / total if total else 0.0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        suffix = f" — {'; '.join(result.reasons)}" if result.reasons else ""
        print(f"[{status}] {result.ticker}{suffix}")
    print(f"\nPass rate: {passed}/{total} ({rate:.0%})")
    if rate < THRESHOLD:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
