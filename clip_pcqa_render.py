"""pytorch3d's point rasteriser, as run with bin_size=0, and its alpha compositing, as
get_projections.py renders with them, in plain torch (pytorch3d 0.7.9:
csrc/rasterize_points/rasterize_points.cu, csrc/compositing/alpha_composite.cu).

On points in view coordinates (x, y, z) from clip_pcqa_camera, x and y also the normalised
device coordinates and z the depth: in an S x S image the pixel of row i and column j has its
centre at (1 - (2j + 1)/S, 1 - (2i + 1)/S), and a point with z >= 0 is drawn in it when

    d^2 = dx^2 + dy^2 < r^2,   dx = x_j - x,   dy = y_i - y,

and the pixel keeps the K drawn points of least depth, nearest first. Its colour is

    c = sum_k c_k a_k prod_(l < k) (1 - a_l),   a_k = 1 - d_k^2 / r^2.

Every float32 operation is pytorch3d's, in its order; d^2 is fma(dy, dy, dx * dx) as its CUDA
kernel is compiled (clip_pcqa_fma). Equal depths go by point index, the fork's convention:
pytorch3d with bin_size=0 orders them by the slots of its 10-slot queue, its binned rasteriser,
which get_projections.py runs, as its atomics happen to.
"""

import math

import torch
from torch import Tensor

from clip_pcqa_fma import fma

# get_projections.py's rasterisation: 1024 x 1024 pixels, discs of radius 0.003 in normalised
# device coordinates, ten points per pixel.
IMAGE_SIZE = 1024
RADIUS = 0.003
POINTS_PER_PIXEL = 10
# A disc reaches pixels whose centres lie within RADIUS * IMAGE_SIZE / 2 pixels of the point.
_REACH = math.ceil(RADIUS * IMAGE_SIZE / 2)


def rasterize(view: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """For every pixel the indices, depths and squared distances of its nearest drawn points,
    nearest first, -1 past the last, [S, S, K] each.

    Args:
        view: Points in view coordinates, [N, 3], float32.
    """
    size, device, z = IMAGE_SIZE, view.device, view[:, 2]
    pixels, points, dist2 = _drawn(view)
    # The pairs run in point order; two stable sorts give pixel, then depth, then point.
    # z + 0.0 turns -0.0 into 0.0, which pytorch3d's comparisons hold equal.
    order = torch.argsort(z[points] + 0.0, stable=True)
    order = order[torch.argsort(pixels[order], stable=True)]
    pixels, points, dist2 = pixels[order], points[order], dist2[order]
    rank = torch.arange(len(pixels), device=device) - torch.searchsorted(pixels, pixels)
    kept = rank < POINTS_PER_PIXEL
    slots = pixels[kept] * POINTS_PER_PIXEL + rank[kept]
    shape = (size, size, POINTS_PER_PIXEL)
    idx = torch.full(shape, -1, dtype=torch.long, device=device).view(-1)
    zbuf = torch.full(shape, -1.0, device=device).view(-1)
    dists = torch.full(shape, -1.0, device=device).view(-1)
    idx[slots], zbuf[slots], dists[slots] = points[kept], z[points[kept]], dist2[kept]
    fragments = (idx.view(shape), zbuf.view(shape), dists.view(shape))
    return fragments


def composite(idx: Tensor, dists: Tensor, colours: Tensor) -> Tensor:
    """pytorch3d's alpha compositing of every pixel's points, front to back, [S, S, C].

    Args:
        idx: The points every pixel keeps, [S, S, K], -1 past the last.
        dists: Their squared distances to the pixel, [S, S, K].
        colours: The points' colours, [N, C].
    """
    alphas = 1 - dists / (RADIUS * RADIUS)
    image = torch.zeros(colours.shape[1], *idx.shape[:2], device=colours.device)
    remaining = torch.ones(idx.shape[:2], device=colours.device)
    for k in range(idx.shape[2]):
        drawn = idx[..., k] >= 0
        alpha = alphas[..., k]
        colour = colours[idx[..., k].clamp(min=0)].permute(2, 0, 1)
        image = image + torch.where(drawn, colour * remaining * alpha, 0.0)
        remaining = torch.where(drawn, remaining * (1 - alpha), remaining)
    return image.permute(1, 2, 0)


def _drawn(view: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Every (pixel, point) pair the point is drawn in, the pixel as row * S + column, with
    its squared distance, in point order; a disc can reach the pixels within _REACH rows and
    columns of the one nearest its centre.

    Args:
        view: Points in view coordinates, [N, 3], float32.
    """
    x, y, z = view.unbind(1)
    size, device = IMAGE_SIZE, view.device
    offsets = torch.arange(-_REACH, _REACH + 1, device=device)
    row_offsets, column_offsets = (
        grid.reshape(1, -1) for grid in torch.meshgrid(offsets, offsets, indexing="ij")
    )
    rows = (torch.round(((1 - y) * size - 1) / 2).long()[:, None] + row_offsets).reshape(-1)
    columns = (torch.round(((1 - x) * size - 1) / 2).long()[:, None] + column_offsets).reshape(-1)
    points = torch.arange(len(view), device=device).repeat_interleave(row_offsets.numel())
    inside = (rows >= 0) & (rows < size) & (columns >= 0) & (columns < size) & ~(z[points] < 0)
    rows, columns, points = rows[inside], columns[inside], points[inside]
    dx = (1 - (2 * columns + 1) / size).float() - x[points]
    dy = (1 - (2 * rows + 1) / size).float() - y[points]
    dist2 = fma(dy, dy, dx * dx)
    radius = torch.tensor(RADIUS, device=device)
    drawn = dist2 < radius * radius
    pairs = ((rows * size + columns)[drawn], points[drawn], dist2[drawn])
    return pairs
