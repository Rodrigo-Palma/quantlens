"""Small-sample statistics for eval reporting: Wilson intervals and power."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_Z95 = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True)
class Rate:
    """A binomial proportion with its 95% Wilson score interval."""

    successes: int
    n: int
    low: float
    high: float

    @property
    def value(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def __str__(self) -> str:
        return f"{self.successes}/{self.n} = {self.value:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson(successes: int, n: int, z: float = _Z95) -> Rate:
    """Wilson score interval; the 0/n and n/n edges are exact, not float residue."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be within [0, n]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == n else min(1.0, centre + margin)
    return Rate(successes, n, low, high)


def mde_two_proportions(p: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest difference from ``p`` detectable between two arms of size ``n``.

    Normal approximation of a two-sided, two-sample test of proportions with the
    variance evaluated at ``p``. ``p`` is clamped away from 0 and 1 so a perfect
    arm still yields a finite, conservative answer.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    return (z_alpha + z_power) * math.sqrt(2 * p * (1 - p) / n)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    ``only_a`` is the number of cases system A passed and B failed; ``only_b`` the
    reverse. Concordant pairs carry no information about the difference.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
