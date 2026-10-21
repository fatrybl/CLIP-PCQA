"""CLIP-PCQA's score of one point cloud in memory: the released pipeline as one function of a
cloud and a seed. The six projections of get_projections.py (clip_pcqa_project), the test-time
preprocessing of cross_validation.py and the released network give

    Q = sum_k p_k q_k,   p = softmax(100 f . t_k),

with f the image feature, t_k the features of the prompts of the quality levels bad .. excellent
and q the checkpoint's scores of those levels. A view under 224 pixels on a side is cropped at
random, as the release does at test time; the draws come from the CPU generator seeded for the
call, so one cloud and one seed give one score, and the caller's random streams are left as
they were. Matmuls and convolutions run in IEEE float32 whatever the caller has set.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from clip_pcqa_network import NETWORK, ROOT
from clip_pcqa_project import project

# The released checkpoints, each trained on a whole database, and the scores of its quality
# levels there (cross_validation.py).
CHECKPOINTS = {
    "SJTU": ("total_ckpts/SJTU_best_model.pth", [2.0, 4.0, 6.0, 8.0, 10.0]),
    "LS_PCQA_part": ("total_ckpts/LS_PCQA_part_best_model.pth", [1.0, 2.0, 3.0, 4.0, 5.0]),
    "BASICS": ("total_ckpts/BASICS_best_model.pth", [0.0, 1.0, 2.0, 3.0, 4.0]),
}
QUALITY_CLASSES = ["bad", "poor", "fair", "good", "excellent"]
CROP = 224
# cross_validation.py's arguments: 16 learned context tokens per class, around its word.
_SETTINGS = SimpleNamespace(
    n_ctx=16,
    class_token_position="middle",
    csc=True,
    ctx_init="",
    prec="fp32",
    subsample_classes="all",
)
_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# nn.DataParallel, which trained the network, prefixes every key of its state.
_PARALLEL_PREFIX = "module."


class CloudScorer:
    """The released CLIP-PCQA on one device."""

    def __init__(self, device: torch.device, checkpoint: str) -> None:
        """Load the released network and one of its checkpoints.

        Args:
            device: Where the cloud is rendered and the network runs.
            checkpoint: Key of CHECKPOINTS, the database the network was trained on.
        """
        path, scores = CHECKPOINTS[checkpoint]
        # CLIP's modules draw their initial weights, which the checkpoint replaces.
        with torch.random.fork_rng([device] if device.type == "cuda" else []):
            network = NETWORK.CLIPPCQA_Net(device, _SETTINGS, scores, QUALITY_CLASSES)
        state = torch.load(ROOT / path, map_location=device)
        network.load_state_dict(
            {key.replace(_PARALLEL_PREFIX, ""): value for key, value in state.items()}
        )
        self._network = network.to(device).eval()
        self._device = device

    def __call__(self, points: np.ndarray, colours: np.ndarray, seed: int) -> float:
        """The released score of one cloud.

        Args:
            points: Coordinates, N x 3.
            colours: Colours, N x 3, in 0..1, or in 0..255 when any exceeds 1.
            seed: Seed of the random crops.
        """
        with _ieee_float32():
            views = project(points, colours, self._device)
        with torch.random.fork_rng([]):
            torch.default_generator.manual_seed(seed)
            pairs = [view_tensors(colour, depth) for colour, depth in views]
        colour = torch.stack([pair[0] for pair in pairs])[None].to(self._device)
        depth = torch.stack([pair[1] for pair in pairs])[None].to(self._device)
        with torch.no_grad(), _ieee_float32():
            _, _, quality, _, _ = self._network(colour, depth)
        return float(quality)


def view_tensors(colour: np.ndarray, depth: np.ndarray) -> tuple[Tensor, Tensor]:
    """One view's network inputs, cross_validation.py's code for one view: a view under the
    crop size on a side is cropped to a square of its short side at random, the colour and
    the depth map each by a draw of its own, and resized; then the central crop, normalised.

    Args:
        colour: The colour map, uint8 RGB.
        depth: The depth map, uint8 in three equal channels.
    """
    read_frame = Image.fromarray(colour)
    depth_frame = Image.fromarray(depth)
    width, height = read_frame.size
    if width < CROP and width < height:
        read_frame = transforms.RandomCrop(width)(read_frame)
        read_frame = transforms.Resize((CROP, CROP))(read_frame)
        depth_frame = transforms.RandomCrop(width)(depth_frame)
        depth_frame = transforms.Resize((CROP, CROP))(depth_frame)
    elif height < CROP:
        read_frame = transforms.RandomCrop(height)(read_frame)
        read_frame = transforms.Resize((CROP, CROP))(read_frame)
        depth_frame = transforms.RandomCrop(height)(depth_frame)
        depth_frame = transforms.Resize((CROP, CROP))(depth_frame)
    tensors = (_TRANSFORM(read_frame), _TRANSFORM(depth_frame))
    return tensors


@contextmanager
def _ieee_float32() -> Iterator[None]:
    """Float32 matmuls and convolutions in IEEE precision for the block, restoring the
    caller's settings after it."""
    matmuls = torch.get_float32_matmul_precision()
    convolutions = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmuls)
        torch.backends.cudnn.allow_tf32 = convolutions
