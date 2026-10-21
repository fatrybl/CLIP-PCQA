"""get_projections.py's six projections of one cloud without pytorch3d: the cloud centred on
its centroid and scaled so that its farthest point lies at distance 1,

    X' = (X - mean X) / max_i |X_i - mean X|,

rendered from six cameras at distance 1 (clip_pcqa_render), its colour map composited on
white and its depth map z / max z, both times 255 and truncated to uint8, then both cropped to
the box of the drawn pixels as the released crop does it, the box's last row and column left
out. The maps are those get_projections.py writes, read back as the dataset reads them: RGB
colour, and depth in three equal channels.
"""

import numpy as np
import torch
from torch import Tensor

from clip_pcqa_camera import look_at_view_transform, view_points
from clip_pcqa_render import composite, rasterize

# get_projections.py's six views, (elevation, azimuth) in degrees: four around the y axis,
# then from +y and -y.
VIEWS = ((0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0))
# Background colour, white, and the levels the maps are scaled to.
_WHITE = 1.0
_LEVELS = 255


def project(
    points: np.ndarray, colours: np.ndarray, device: torch.device
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The six colour and depth maps of a cloud, uint8, height x width x 3 each.

    Args:
        points: Coordinates, N x 3.
        colours: Colours, N x 3, in 0..1, or in 0..255 when any exceeds 1, as the release
            reads them.
        device: Where the cloud is rendered.
    """
    verts = torch.Tensor(normalize_verts(points)).to(device)
    rgb = torch.Tensor(colours).to(device)
    if torch.sum(rgb > 1) >= 1:
        rgb = rgb / 255
    maps = [_view_maps(verts, rgb, elev, azim) for elev, azim in VIEWS]
    return maps


def normalize_verts(verts: np.ndarray) -> np.ndarray:
    """get_projections.normalize_verts: the cloud centred and scaled into the unit sphere.

    Args:
        verts: Coordinates, N x 3.
    """
    centroid = np.mean(verts, axis=0)
    verts = verts - centroid
    max_length = np.max(np.sqrt(np.sum(verts**2, axis=1)))
    verts_normalized = verts / max_length
    return verts_normalized


def _view_maps(verts: Tensor, rgb: Tensor, elev: int, azim: int) -> tuple[np.ndarray, np.ndarray]:
    """One view's colour and depth maps, cropped. The release's float32 arithmetic treats the
    pixels one by one, so it runs on the crop, on the GPU, where division, multiplication and
    truncation round as numpy's do.

    Args:
        verts: Normalised coordinates, N x 3, float32.
        rgb: Colours in 0..1, N x 3, float32.
        elev: Elevation of the camera in degrees.
        azim: Its azimuth in degrees.
    """
    R, T = look_at_view_transform(elev, azim)
    idx, zbuf, dists = rasterize(view_points(verts, R.to(verts.device), T.to(verts.device)))
    drawn = zbuf[..., 0] != -1
    rows, columns = torch.nonzero(drawn, as_tuple=True)
    box = np.s_[int(rows.min()) : int(rows.max()), int(columns.min()) : int(columns.max())]
    depth = torch.where(drawn, zbuf[..., 0], 0.0)
    max_depth = depth[depth != 0.0].max()
    depth_map = (depth[box] / max_depth * _LEVELS).to(torch.uint8)
    image = torch.where(drawn[box][..., None], composite(idx, dists, rgb)[box], _WHITE)
    colour_map = (image * _LEVELS).to(torch.uint8)
    maps = (colour_map.cpu().numpy(), depth_map[..., None].expand(-1, -1, 3).cpu().numpy())
    return maps
