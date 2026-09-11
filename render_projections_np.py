"""Dependency-free replacement for get_projections.py (which needs pytorch3d).

Replicates PyTorch3D's FoVOrthographicCameras + PointsRasterizer + AlphaCompositor as used
by the original script: unit-sphere normalisation, six look_at views at distance 1,
1024x1024 image, point radius 0.003 NDC, 10 points per pixel, white background, depth
normalised by its maximum, then a crop to the mask bounding box.
"""
import argparse
import os

import numpy as np
from PIL import Image

RENDER_VIEWS_6 = [(0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0)]


def normalize_verts(verts):
    centroid = np.mean(verts, axis=0)
    verts = verts - centroid
    max_length = np.max(np.sqrt(np.sum(verts ** 2, axis=1)))
    return verts / max_length


def _normalize(v, eps=1e-5):
    # torch.nn.functional.normalize semantics: v / max(||v||, eps)
    return v / max(float(np.linalg.norm(v)), eps)


def look_at_view_transform(dist, elev, azim):
    """float32 replica of pytorch3d.renderer.look_at_view_transform (at=0, up=+y)."""
    f = np.float32
    elev = f(np.pi / 180.0) * f(elev)
    azim = f(np.pi / 180.0) * f(azim)
    cam = np.array([dist * np.cos(elev) * np.sin(azim), dist * np.sin(elev), dist * np.cos(elev) * np.cos(azim)], dtype=f)
    at = np.zeros(3, f)
    up = np.array([0, 1, 0], f)
    z_axis = _normalize(at - cam).astype(f)
    x_axis = _normalize(np.cross(up, z_axis)).astype(f)
    y_axis = _normalize(np.cross(z_axis, x_axis)).astype(f)
    if np.all(np.isclose(x_axis, 0.0, atol=5e-3)):
        x_axis = _normalize(np.cross(y_axis, z_axis)).astype(f)
    R = np.stack([x_axis, y_axis, z_axis], axis=0).T  # X_cam = X_world @ R + T
    T = -(R.T @ cam)
    return R.astype(f), T.astype(f)


def rasterize(pts_cam, rgb, image_size=1024, radius=0.003, points_per_pixel=10):
    """Return (rgb image float [S,S,3] composited over black, zbuf [S,S] with -1 = empty)."""
    S = image_size
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    col_f = ((1.0 - x) * S - 1.0) / 2.0  # +X NDC points left, +Y up (pytorch3d convention)
    row_f = ((1.0 - y) * S - 1.0) / 2.0
    r_px = radius * S / 2.0
    reach = int(np.ceil(r_px))
    offs = np.arange(-reach, reach + 1)
    dy, dx = np.meshgrid(offs, offs, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    ci = (np.round(col_f)[:, None] + dx[None, :]).astype(np.int64)
    ri = (np.round(row_f)[:, None] + dy[None, :]).astype(np.int64)
    d2_px = (ci - col_f[:, None]) ** 2 + (ri - row_f[:, None]) ** 2
    ok = (d2_px <= r_px ** 2) & (ci >= 0) & (ci < S) & (ri >= 0) & (ri < S) & (z[:, None] >= 0)  # pytorch3d skips points behind the camera
    pt = np.repeat(np.arange(len(x)), len(dx)).reshape(len(x), len(dx))[ok]
    pix = (ri * S + ci)[ok]
    d2_ndc = d2_px[ok] * (2.0 / S) ** 2
    zz = z[pt]
    order = np.lexsort((zz, pix))
    pt, pix, d2_ndc, zz = pt[order], pix[order], d2_ndc[order], zz[order]
    first = np.r_[True, pix[1:] != pix[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(pix)), 0))
    rank = np.arange(len(pix)) - start
    keep = rank < points_per_pixel
    pt, pix, d2_ndc, zz, rank, first = pt[keep], pix[keep], d2_ndc[keep], zz[keep], rank[keep], first[keep]
    alpha = 1.0 - d2_ndc / radius ** 2
    log1m = np.log(np.clip(1.0 - alpha, 1e-10, None))
    cs = np.cumsum(log1m)
    start = np.maximum.accumulate(np.where(first, np.arange(len(pix)), 0))
    # exclusive prefix within group: sum of log1m of earlier entries in the same pixel
    grp_offset = np.where(start > 0, cs[start - 1], 0.0)
    excl = (cs - log1m) - grp_offset
    weight = alpha * np.exp(excl)
    img = np.zeros((S * S, 3), np.float64)
    for c in range(3):
        img[:, c] = np.bincount(pix, weights=weight * rgb[pt, c], minlength=S * S)
    zbuf = np.full(S * S, -1.0)
    zbuf[pix[rank == 0]] = zz[rank == 0]
    return img.reshape(S, S, 3), zbuf.reshape(S, S)


def render_views(points, rgb, views=RENDER_VIEWS_6, image_size=1024, radius=0.003, points_per_pixel=10):
    """Return [(colour_crop uint8 RGB HxWx3, depth_crop uint8 HxWx3)] per view, like get_projections.py."""
    verts = normalize_verts(points[:, :3].astype(np.float64)).astype(np.float32)
    rgb = np.asarray(rgb, np.float32)
    if np.sum(rgb > 1) >= 1:
        rgb = rgb / 255.0
    out = []
    for elev, azim in views:
        R, T = look_at_view_transform(1.0, elev, azim)
        pts_cam = verts @ R + T
        img, zbuf = rasterize(pts_cam.astype(np.float64), rgb.astype(np.float64), image_size, radius, points_per_pixel)
        mask = zbuf != -1
        depth = np.where(mask, zbuf, 0.0)
        max_depth = depth[mask].max()
        depth_u8 = (depth / max_depth * 255).astype(np.uint8)
        img[~mask] = 1.0
        img_u8 = (img * 255).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        out.append((img_u8[y0:y1, x0:x1], np.repeat(depth_u8[y0:y1, x0:x1, None], 3, axis=2)))
    return out


def load_points(path):
    import open3d as o3d
    if path.endswith(".npy"):
        a = np.load(path)
        if a.shape[1] < 6:
            raise SystemExit(f"{path} has {a.shape[1]} columns, expected x y z r g b; "
                             "for x y z intensity run tools/npy_intensity_to_ply.py first")
        return a[:, :3], a[:, 3:6]
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        return pts, np.asarray(pcd.colors)
    raise SystemExit(f"{path} has no colours")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".ply with colours or .npy with xyzrgb columns")
    ap.add_argument("--out_color", required=True)
    ap.add_argument("--out_depth", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_color, exist_ok=True)
    os.makedirs(a.out_depth, exist_ok=True)
    pts, rgb = load_points(a.input)
    stem = os.path.splitext(os.path.basename(a.input))[0]
    for k, (img, dep) in enumerate(render_views(pts, rgb)):
        Image.fromarray(img).save(os.path.join(a.out_color, f"{stem}_view_{k}.png"))
        Image.fromarray(dep).save(os.path.join(a.out_depth, f"{stem}_view_{k}.png"))
    print("wrote 6 colour and 6 depth views for", stem)
