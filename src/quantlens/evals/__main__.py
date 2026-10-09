"""Run every offline eval, print the report and fail on a regression.

Usage: ``python -m quantlens.evals`` (``make evals``).
"""

from __future__ import annotations

import sys

from quantlens.evals import faithfulness, guardrail, retrieval
from quantlens.evals.gates import check


def main() -> int:
    faith_scores = faithfulness.run()
    guard_scores = guardrail.run()
    lines = faithfulness.report(faith_scores)
    lines += guardrail.report(guard_scores)
    lines += retrieval.report()
    measured = {
        **faithfulness.measured(faith_scores),
        **guardrail.measured(guard_scores),
        **retrieval.measured(),
    }
    failures = check(measured)
    print("\n".join(lines))
    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll eval gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
