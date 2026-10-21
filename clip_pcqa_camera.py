"""pytorch3d's look-at camera as get_projections.py places it, in plain torch: its rotation R
and translation T for a camera at distance 1 looking at the origin with up +y, and its
world-to-view transform X_view = X R + T, in pytorch3d's batched homogeneous product
(pytorch3d 0.7.9: renderer/cameras.py, transforms/transform3d.py). With the orthographic
projection get_projections.py uses, x and y of X_view are also the normalised device
coordinates and z the depth. Every float32 operation is pytorch3d's, in its order.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# pytorch3d's look-at constants: the camera's distance and up vector, the epsilon of its
# normalisations, and the tolerance within which it takes the up vector as parallel to the view.
_DISTANCE = 1.0
_UP = (0.0, 1.0, 0.0)
_NORMALIZE_EPS = 1e-5
_PARALLEL_ATOL = 5e-3
_RADIANS_PER_DEGREE = math.pi / 180.0


def look_at_view_transform(elev: float, azim: float) -> tuple[Tensor, Tensor]:
    """pytorch3d's look_at_view_transform at distance 1 towards the origin, up +y, angles in
    degrees: R, [1, 3, 3], and T, [1, 3], of X_view = X R + T, float32 on the CPU.

    Args:
        elev: Elevation of the camera above the plane y = 0.
        azim: Azimuth of the camera from +z towards +x.
    """
    dist, elev_rad, azim_rad = (
        torch.tensor([value], dtype=torch.float32) for value in (_DISTANCE, elev, azim)
    )
    elev_rad = _RADIANS_PER_DEGREE * elev_rad
    azim_rad = _RADIANS_PER_DEGREE * azim_rad
    x = dist * torch.cos(elev_rad) * torch.sin(azim_rad)
    y = dist * torch.sin(elev_rad)
    z = dist * torch.cos(elev_rad) * torch.cos(azim_rad)
    at = torch.zeros(1, 3)
    up = torch.tensor([_UP])
    camera = torch.stack([x, y, z], dim=1).view(-1, 3) + at
    z_axis = F.normalize(at - camera, eps=_NORMALIZE_EPS)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=_NORMALIZE_EPS)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=_NORMALIZE_EPS)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=_PARALLEL_ATOL).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=_NORMALIZE_EPS)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    R = R.transpose(1, 2)
    T = -torch.bmm(R.transpose(1, 2), camera[:, :, None])[:, :, 0]
    return R, T


def view_points(points: Tensor, R: Tensor, T: Tensor) -> Tensor:
    """Points in view coordinates by pytorch3d's composed transform, [X 1] M with
    M = [[R, 0], [T, 1]], in its batched product, float32.

    Args:
        points: World coordinates, [N, 3].
        R: The camera's rotation, [1, 3, 3], on the points' device.
        T: Its translation, [1, 3], on the points' device.
    """
    rotate = torch.eye(4, device=points.device).view(1, 4, 4).repeat(1, 1, 1)
    rotate[:, :3, :3] = R
    translate = torch.eye(4, device=points.device).view(1, 4, 4).repeat(1, 1, 1)
    translate[:, 3, :3] = T
    ones = torch.ones(1, len(points), 1, device=points.device)
    homogeneous = torch.cat([points[None], ones], dim=2).bmm(rotate.bmm(translate))
    view = homogeneous[0, :, :3] / homogeneous[0, :, 3:]
    return view
