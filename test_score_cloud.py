"""The fork's projections and scorer against the release's files: the maps clip_pcqa_project
returns are those get_projections.py writes and crops, read back as cross_validation.py reads
them; one cloud and one seed give one score, and the caller's random streams stay as they were.

The release's crop runs from the source of get_projections.py itself, whose module renders a
dataset with pytorch3d when imported. Run from the fork's root:
``python -m pytest test_score_cloud.py`` (GPU, released weights).
"""

import ast
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from clip_pcqa_camera import look_at_view_transform, view_points
from clip_pcqa_project import VIEWS, normalize_verts, project
from clip_pcqa_render import composite, rasterize
from score_cloud import CloudScorer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the scorer renders on the GPU"
)

_SCRIPT = Path(__file__).with_name("get_projections.py")
_SEED = 2026
_POINTS = 40_000
_EXTENT_M = (40.0, 40.0, 4.0)


@pytest.fixture
def cloud() -> tuple[np.ndarray, np.ndarray]:
    """A scan-like slab of points and their grey colour in 0..1."""
    rng = np.random.default_rng(_SEED)
    grey = np.repeat(rng.uniform(size=(_POINTS, 1)), 3, axis=1)
    slab = (rng.normal(size=(_POINTS, 3)) * _EXTENT_M, grey)
    return slab


def test_maps_are_the_released_files(cloud: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    """Each view's maps, written, cropped and read back with the release's statements, equal
    the fork's.

    Args:
        cloud: Coordinates and colours.
        tmp_path: Where the release's files are written.
    """
    points, colours = cloud
    crop = _released("find_min_bounding_rect", "crop_image_with_mask")["crop_image_with_mask"]
    verts = torch.Tensor(normalize_verts(points)).cuda()
    rgb = torch.Tensor(colours).cuda()
    for (elev, azim), maps in zip(VIEWS, project(points, colours, torch.device("cuda"))):
        R, T = look_at_view_transform(elev, azim)
        idx, zbuf, dists = rasterize(view_points(verts, R.cuda(), T.cuda()))
        depth = zbuf[None, ..., 0].cpu()
        binary_mask_data = (depth != -1).float().squeeze().numpy()
        Image.fromarray((binary_mask_data * 255).astype(np.uint8)).save(tmp_path / "mask.png")
        depth = torch.where(depth == -1, torch.tensor(0.0), depth)
        max_depth = torch.max(depth[depth != 0.0]).numpy()
        depth_data = depth.squeeze().numpy()
        Image.fromarray((depth_data / max_depth * 255).astype(np.uint8)).save(
            tmp_path / "depth.png"
        )
        img_array = composite(idx, dists, rgb).cpu().numpy()
        img_array[binary_mask_data == 0] = 1
        img_array = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(tmp_path / "colour.png"), img_array)
        mask_image = cv2.imread(str(tmp_path / "mask.png"))
        for name, ours in zip(("colour", "depth"), maps):
            cropped = crop(cv2.imread(str(tmp_path / f"{name}.png")), mask_image)
            cv2.imwrite(str(tmp_path / f"{name}_crop.png"), cropped)
            read = np.asarray(Image.open(tmp_path / f"{name}_crop.png").convert("RGB"))
            np.testing.assert_array_equal(read, ours)


def test_one_cloud_and_one_seed_give_one_score(cloud: tuple[np.ndarray, np.ndarray]) -> None:
    """The same score twice, and the caller's random streams left as they were.

    Args:
        cloud: Coordinates and colours.
    """
    scorer = CloudScorer(torch.device("cuda"), "LS_PCQA_part")
    states = (torch.random.get_rng_state(), torch.cuda.get_rng_state())
    first, again = scorer(*cloud, _SEED), scorer(*cloud, _SEED)
    assert first == again
    assert torch.equal(states[0], torch.random.get_rng_state())
    assert torch.equal(states[1], torch.cuda.get_rng_state())


def _released(*names: str) -> dict[str, Callable[..., np.ndarray]]:
    """Functions of get_projections.py, executed from its source in one namespace, where they
    find one another and the modules they use.

    Args:
        names: The functions to take.
    """
    tree = ast.parse(_SCRIPT.read_text())
    taken = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace: dict[str, Callable[..., np.ndarray]] = {"np": np, "cv2": cv2}
    module = ast.Module(body=taken, type_ignores=[])
    exec(compile(module, str(_SCRIPT), "exec"), namespace)
    return namespace
