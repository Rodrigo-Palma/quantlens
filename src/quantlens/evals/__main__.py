"""Run every offline eval, print the report and fail on a regression.

Usage: ``python -m quantlens.evals`` (``make evals``). ``--update-gates`` rewrites
``data/gates.json`` with the measured counts (commit it on its own).
"""

from __future__ import annotations

import argparse
import sys

from quantlens.evals import checker_audit, faithfulness, gates, guardrail, retrieval


def measure() -> tuple[list[str], dict[str, int]]:
    faith_scores = faithfulness.run()
    audit_scores = checker_audit.run()
    guard_scores = guardrail.run()
    lines = faithfulness.report(faith_scores)
    lines += checker_audit.report(audit_scores)
    lines += guardrail.report(guard_scores)
    lines += retrieval.report()
    measured = {
        **faithfulness.measured(faith_scores),
        **checker_audit.measured(audit_scores),
        **guardrail.measured(guard_scores),
        **retrieval.measured(),
    }
    return lines, measured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-gates", action="store_true", help="record the measured counts as the gates"
    )
    args = parser.parse_args(argv)
    lines, measured = measure()
    print("\n".join(lines))
    if args.update_gates:
        path = gates.write(gates.updated(measured, gates.load_bounds()))
        print(f"\nGates rewritten from this run: {path}")
        return 0
    failures = gates.check(measured)
    if failures:
        print("\nGATE FAILED (a recorded count got worse):\n  " + "\n  ".join(failures))
        return 1
    better = gates.improvements(measured)
    if better:
        print("\nBetter than recorded (tighten with --update-gates):\n  " + "\n  ".join(better))
    print("\nAll eval gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
