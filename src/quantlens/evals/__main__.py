"""Run every offline eval, print the report and fail on a regression.

Usage: ``python -m quantlens.evals`` (``make evals``).
"""

from __future__ import annotations

import sys

from quantlens.evals import guardrail, retrieval
from quantlens.evals.gates import check


def main() -> int:
    lines: list[str] = []
    lines += retrieval.report()
    guard_scores = guardrail.run()
    lines += guardrail.report(guard_scores)
    failures = check({**retrieval.measured(), **guardrail.measured(guard_scores)})
    print("\n".join(lines))
    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll eval gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
