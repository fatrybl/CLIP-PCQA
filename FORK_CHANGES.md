# How this fork differs from the original CLIP-PCQA code

**Original:** [Olivialyt/CLIP-PCQA](https://github.com/Olivialyt/CLIP-PCQA), commit `1de267d`
(2025-06-16), the code of *CLIP-PCQA: Exploring Subjective-Aligned Vision-Language Modeling for
Point Cloud Quality Assessment* (Liu et al., AAAI 2025).
**Fork:** [fatrybl/CLIP-PCQA](https://github.com/fatrybl/CLIP-PCQA), branch `vape-baselines`.
`git diff 1de267d` shows every change.

The fork serves as a baseline in the comparison of the VAPE project, where the released network
scores a LiDAR point cloud. The method is unchanged: the network and its weights, the
projections, the test-time preprocessing with its random crops, and the score are the released
code's. The fork adds a scorer that runs the released pipeline on a cloud in memory, renders
without pytorch3d, and seeds the random crops.

## 1. Rendering without pytorch3d

- **Why:** `get_projections.py` renders with pytorch3d, which has no official build for torch 2.9
  and CUDA 13. A community wheel (`pytorch3d-0.7.9+pt2.9.1cu130`,
  [MiroPsota/torch_packages_builder](https://github.com/MiroPsota/torch_packages_builder)) runs,
  but the binned rasteriser `get_projections.py` uses is nondeterministic on LiDAR clouds. On a
  62 522-point ONCE scan the default cap of points per bin drops points: pytorch3d reports it
  only by a printf from the GPU on stdout ("Bin size was too small in the coarse rasterization
  phase", 646 times for the six views), not by a Python warning or error, and between two calls
  the depth buffer differs at about 2 000 pixels of one view. Points of equal depth are ordered
  as its atomics happen to (900 to 1 350 pixels of that view, even with the cap raised). Its
  deterministic mode (`bin_size=0`) takes 1.1 to 3.6 s for the six views of a ONCE union,
  against 40 ms for the network.
- **What:** `clip_pcqa_render.py` reimplements in plain torch the parts of pytorch3d that
  `get_projections.py` uses: `look_at_view_transform`, the world-to-view transform, the
  orthographic projection, the point rasteriser as run with `bin_size=0`, and alpha compositing,
  every float32 operation in pytorch3d's order (read from pytorch3d 0.7.9's `cameras.py`,
  `transform3d.py`, `rasterize_points.cu` and `alpha_composite.cu`). The squared distance is
  `fma(dy, dy, dx * dx)`, as nvcc compiles the kernel's `dx * dx + dy * dy`; `clip_pcqa_fma.py`
  rounds it once, correcting the double rounding a float64 emulation can make. Points of equal
  depth go by point index, a convention of the fork: pytorch3d's naive kernel keeps a 10-slot
  queue in point order, overwrites the first slot holding the largest depth when a nearer point
  arrives, and sorts the slots by depth, so its order of equal depths follows the slots.
- **Verified** against pytorch3d 0.7.9 with `bin_size=0` (`test_render.py`, and on the 60 unions
  of VAPE's paper ladder, ten ONCE samples at three levels at the true and the start pose, 360
  views): the cameras, the view coordinates and the depth buffers, all ten slots of every pixel,
  are bitwise equal; the squared distances are bitwise equal on every entry where both keep the
  same point; the colours are bitwise equal at every pixel that keeps the same points. A pixel
  keeps other points only among points of equal depth and only where more than ten points are
  drawn in it, 2.0 % of the covered pixels of those views; a literal emulation of the naive
  kernel's queue reproduces pytorch3d's lists at all 20 793 pixels probed on two ONCE unions,
  every differing one included. The depth maps equal the files `get_projections.py` writes in
  that mode in all 360 views; 11 282 colour pixels differ, in 159 views, every one at such a
  pixel. The fused multiply-add is exact against rational arithmetic on a billion realistic
  cases and on constructed cases where a float64 emulation rounds twice wrongly (`test_fma.py`).
  Six views take 46 to 72 ms on an RTX 3090.
- **Effect on the score** (LS-PCQA weights): on those 60 unions, with the 21 crop seeds 2026 and
  0 to 19, the fork's score lies a median 1·10⁻⁵ and at most 1.8·10⁻³ from that of the
  authors' code (`get_projections.py` with `bin_size=0`, then `utils/CLIPPCQA_Dataset.py` and
  the network) on a scale of 1 to 5, the largest where a random crop of a side view takes in
  pixels of equal depth. The authors' code as released scores one union and seed up to 8·10⁻²
  apart in two runs (median 2.5·10⁻³) and up to 0.27 from its deterministic mode. On the WPC
  cloud `bag_level_7.ply` (MM-PCQA's demonstration sample: coloured 0 to 255, every view over
  224 pixels) the fork's score equals the authors' code's bit for bit in both modes.

## 2. A scorer of one cloud

- `score_cloud.py`: `CloudScorer(device, checkpoint)(points, colours, seed)` returns the released
  score `Q = sum_k p_k q_k`. `clip_pcqa_project.py` makes the six maps as `get_projections.py`
  makes them (normalisation, six views, colour on white, depth `z / max z`, truncation to uint8,
  the release's crop); `view_tensors` is `cross_validation.py`'s preprocessing of one view; the
  network is the released one. The random crops are drawn from the CPU generator seeded for the
  call, inside a fork of the random state, so one cloud and one seed give one score and the
  caller's streams stay as they were. Matmuls and convolutions run in IEEE float32.
- `clip_pcqa_network.py` loads the fork's `clip` package and the network under names of their
  own, since `clip` and `models` are common top-level names in a host process.
- **Verified:** the maps equal the files `get_projections.py` writes and crops, read back as
  `cross_validation.py` reads them (`test_score_cloud.py`); on eight ONCE evaluations the score
  of the maps in memory equals the score through those files bit for bit; one cloud and one seed
  give one score.

## 3. Kept as released

- The network and the three released checkpoints (SJTU-PCQA, LS-PCQA Part I, BASICS) with their
  scores of the five quality levels; VAPE uses LS-PCQA Part I, the best across databases in the
  paper's Table 2.
- The code where it differs from the paper: the logits are scaled by `exp(logit_scale) = 100`
  (eq. 10 has no temperature); the averaged image feature is not normalised again, so the logit
  is a dot product, not eq. 9's cosine; colour and depth tokens are normalised before they are
  averaged (eq. 5 sums them raw).
- `cross_validation.py`'s preprocessing: a view under 224 pixels on a side is cropped at random
  to a square of its short side and resized (also at test time; the dataset's `is_train` is
  unused), the colour and the depth map each by a draw of its own; ImageNet normalisation.
- The crop of `get_projections.py` leaves out the last row and column of the drawn pixels' box.
- The depth map `z / max z` times 255, truncated, background 0; the colour map on white.

## 4. Other changes

- `get_projections.py`: `find_min_bounding_rect` reshapes `cv2.findNonZero`'s points, which
  OpenCV 5 returns as N x 2 and earlier versions as N x 1 x 2; nothing changes for the latter.
- Removed: the bytecode the original authors committed (`clip/__pycache__/`), and this fork's
  earlier `render_projections_np.py`, a numpy renderer never compared with pytorch3d (its discs
  cover 105 more pixels than pytorch3d's in one view of a ONCE scan), with `test_single_ply.py`.
- `.gitignore`: bytecode, build products, logs (the released `logs/` stay tracked) and
  `total_ckpts` (the weights, downloaded from the authors' links).
- Dependency: `ftfy` 6.3.1. The release pins 6.2.0, whose requirement would downgrade `wcwidth`;
  the five prompts tokenize to the same ids under both.

## 5. Tests

`python -m pytest test_fma.py test_render.py test_score_cloud.py` from the repository's root
(GPU, released weights). `test_render.py` needs pytorch3d and is skipped without it; it ran
against the community wheel above, unpacked with `pip install --no-deps --target <dir>` and put
on `PYTHONPATH`.
