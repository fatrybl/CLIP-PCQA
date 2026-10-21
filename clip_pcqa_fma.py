"""The fused multiply-add of float32 tensors, a * b + c rounded once to float32, as pytorch3d's
point rasteriser computes dx * dx + dy * dy once nvcc has contracted it: the product is exact
in float64, and the float64 sum's error term, exact by TwoSum, decides the float32 midpoints
at which rounding the float64 sum again would go the wrong way (test_fma.py builds such cases).
"""

import torch
from torch import Tensor


def fma(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """a * b + c of float32 tensors rounded once to float32, as a fused multiply-add: the
    product is exact in float64 and the sum's error term decides a float32 midpoint.

    Args:
        a: First factor.
        b: Second factor.
        c: Addend.
    """
    product, addend = a.double() * b.double(), c.double()
    total = product + addend
    # TwoSum: product + addend = total + error exactly.
    rest = total - product
    error = (product - (total - rest)) + (addend - rest)
    rounded = total.float()
    other = torch.nextafter(rounded, torch.where(rounded.double() > total, -torch.inf, torch.inf))
    # Where total is the midpoint of rounded and other, the error's sign picks the side.
    midpoint = (rounded.double() + other.double()) / 2 == total
    towards = (error > 0) == (other > rounded)
    result = torch.where(midpoint & (error != 0) & towards, other, rounded)
    return result
