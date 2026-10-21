"""The fork's renderer (clip_pcqa_render.py) against pytorch3d 0.7.9 as get_projections.py
configures it, run deterministically (bin_size=0): the cameras, the view coordinates, the
depths and squared distances of every pixel's points equal pytorch3d's bit for bit, and so do
the colours of every pixel that keeps the same points. With every depth equal, a pixel can
keep other points only among points of equal depth, which the fork orders by point index and
pytorch3d by the slots of its queue (its binned rasteriser, which get_projections.py runs, as
its atomics happen to).

Run from the fork's root: ``python -m pytest test_render.py``; skipped without pytorch3d.
"""

import numpy as np
import pytest
import torch

from clip_pcqa_camera import look_at_view_transform, view_points
from clip_pcqa_project import VIEWS, normalize_verts
from clip_pcqa_render import IMAGE_SIZE, POINTS_PER_PIXEL, RADIUS, composite, rasterize

renderer = pytest.importorskip("pytorch3d.renderer")
structures = pytest.importorskip("pytorch3d.structures")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="get_projections.py renders on the GPU"
)

_SEED = 2026
_POINTS = 30_000
# A scan-like slab; on a 5 cm grid many points share a depth.
_EXTENT_M = (20.0, 20.0, 2.0)
_GRID_M = 0.05


@pytest.mark.parametrize("grid", [None, _GRID_M])
def test_renderer_is_pytorch3ds(grid: float | None) -> None:
    """Every view of a random cloud, and of one on a grid, rendered both ways.

    Args:
        grid: Spacing the coordinates are rounded to, or None.
    """
    rng = np.random.default_rng(_SEED)
    cloud = rng.normal(size=(_POINTS, 3)) * _EXTENT_M
    if grid is not None:
        cloud = np.round(cloud / grid) * grid
    device = torch.device("cuda")
    verts = torch.Tensor(normalize_verts(cloud)).to(device)
    colours = torch.Tensor(rng.uniform(size=(_POINTS, 3))).to(device)
    clouds = structures.Pointclouds(points=[verts], features=[colours])
    settings = renderer.PointsRasterizationSettings(
        image_size=IMAGE_SIZE, radius=RADIUS, points_per_pixel=POINTS_PER_PIXEL, bin_size=0
    )
    for elev, azim in VIEWS:
        R, T = look_at_view_transform(elev, azim)
        expected_R, expected_T = renderer.look_at_view_transform(1, elev, azim)
        assert torch.equal(R, expected_R) and torch.equal(T, expected_T)
        cameras = renderer.FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
        rasterizer = renderer.PointsRasterizer(cameras=cameras, raster_settings=settings)
        view = view_points(verts, R.to(device), T.to(device))
        assert torch.equal(view, rasterizer.transform(clouds).points_padded()[0])
        idx, zbuf, dists = rasterize(view)
        fragments = rasterizer(clouds)
        assert torch.equal(zbuf, fragments.zbuf[0])
        same = idx == fragments.idx[0].long()
        assert torch.equal(dists[same], fragments.dists[0][same])
        weights = 1 - fragments.dists.permute(0, 3, 1, 2) / (RADIUS * RADIUS)
        theirs = renderer.AlphaCompositor()(
            fragments.idx.long().permute(0, 3, 1, 2), weights, colours.T
        )
        lists = same.all(dim=-1)
        image = composite(idx, dists, colours)
        assert torch.equal(image[lists], theirs[0].permute(1, 2, 0)[lists])
