"""clip_pcqa_fma.fma against exact rational arithmetic: a * b + c rounded once to float32,
on constructed cases where the sum, rounded to float64 first, lands on a float32 midpoint (so
that rounding twice is wrong on half of them) and on random cases shaped like the rasteriser's
dy * dy + dx * dx.

Run from the fork's root: ``python -m pytest test_fma.py``.
"""

from fractions import Fraction

import numpy as np
import pytest
import torch

from clip_pcqa_fma import fma

_SEED = 2026
_RANDOM_CASES = 20_000
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
def test_fma_rounds_once(device: str) -> None:
    """Exact on the constructed double-rounding cases, which do double-round, and on random
    rasteriser-like cases.

    Args:
        device: Where fma runs.
    """
    rng = np.random.default_rng(_SEED)
    a, c = _midpoint_cases()
    twice = (a.astype(np.float64) ** 2 + c.astype(np.float64)).astype(np.float32)
    exact = np.array(
        [_exact_float32(Fraction(float(x)) ** 2 + Fraction(float(y))) for x, y in zip(a, c)]
    )
    assert (twice != exact).any()
    on = [torch.from_numpy(values).to(device) for values in (a, c)]
    np.testing.assert_array_equal(fma(on[0], on[0], on[1]).cpu().numpy(), exact)
    dx = (rng.uniform(-1, 1, _RANDOM_CASES) * 0.005).astype(np.float32)
    dy = (rng.uniform(-1, 1, _RANDOM_CASES) * 0.005).astype(np.float32)
    square = dx * dx
    exact = np.array(
        [_exact_float32(Fraction(float(y)) ** 2 + Fraction(float(s))) for y, s in zip(dy, square)]
    )
    on = [torch.from_numpy(values).to(device) for values in (dy, square)]
    np.testing.assert_array_equal(fma(on[0], on[0], on[1]).cpu().numpy(), exact)


def _midpoint_cases() -> tuple[np.ndarray, np.ndarray]:
    """Pairs (a, c) with a = A 2^-23 in [1, 2) and M = (A^2 + j) 2^-46 a float32 midpoint:
    c = j 2^-46 moved by one or two float32 steps makes a^2 + c = M + d with d below half a
    float64 step, so that the float64 sum is M."""
    significands = np.arange(2**23, int(2**23.5), dtype=np.int64)
    a, c = [], []
    for j in range(7, 2**10, 8):
        for significand in significands[((significands * significands + j) % 2**23) == 2**22][:2]:
            for steps, direction in ((1, np.inf), (2, np.inf), (1, -np.inf), (2, -np.inf)):
                addend = np.float32(j * 2.0**-46)
                for _ in range(steps):
                    addend = np.nextafter(addend, np.float32(direction))
                a.append(np.float32(int(significand) / 2**23))
                c.append(addend)
    pairs = (np.array(a, dtype=np.float32), np.array(c, dtype=np.float32))
    return pairs


def _exact_float32(value: Fraction) -> np.float32:
    """The float32 nearest to an exact rational, ties to the even significand.

    Args:
        value: The exact value.
    """
    guess = np.float32(float(value))
    candidates = [
        np.nextafter(guess, np.float32(-np.inf)),
        guess,
        np.nextafter(guess, np.float32(np.inf)),
    ]
    distances = [abs(Fraction(float(candidate)) - value) for candidate in candidates]
    nearest = [
        candidate
        for candidate, distance in zip(candidates, distances)
        if distance == min(distances)
    ]
    even = [candidate for candidate in nearest if int(np.array(candidate).view(np.uint32)) % 2 == 0]
    rounded = nearest[0] if len(nearest) == 1 else even[0]
    return rounded
