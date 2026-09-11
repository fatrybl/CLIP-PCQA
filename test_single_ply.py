"""Score one coloured point cloud with a pretrained CLIP-PCQA checkpoint (CPU or GPU).

Mirrors cross_validation.py: six colour + depth projections, CenterCrop(224) + ImageNet
normalisation, and the score anchors of the dataset the checkpoint was trained on.
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.CLIPPCQA_Net import CLIPPCQA_Net
from render_projections_np import load_points, render_views

CKPTS = {
    "sjtu": ("total_ckpts/SJTU_best_model.pth", [2.0, 4.0, 6.0, 8.0, 10.0]),
    "ls": ("total_ckpts/LS_PCQA_part_best_model.pth", [1.0, 2.0, 3.0, 4.0, 5.0]),
    "basics": ("total_ckpts/BASICS_best_model.pth", [0.0, 1.0, 2.0, 3.0, 4.0]),
}
QUALITY_CLASSES = ["bad", "poor", "fair", "good", "excellent"]


def to_tensor_pair(img_u8, dep_u8, crop_size=224):
    read_frame = Image.fromarray(img_u8).convert("RGB")
    depth_frame = Image.fromarray(dep_u8).convert("RGB")
    width, height = read_frame.size
    if width < crop_size and width < height:
        read_frame = transforms.Resize((crop_size, crop_size))(transforms.RandomCrop(width)(read_frame))
        depth_frame = transforms.Resize((crop_size, crop_size))(transforms.RandomCrop(width)(depth_frame))
    elif height < crop_size:
        read_frame = transforms.Resize((crop_size, crop_size))(transforms.RandomCrop(height)(read_frame))
        depth_frame = transforms.Resize((crop_size, crop_size))(transforms.RandomCrop(height)(depth_frame))
    tf = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return tf(read_frame), tf(depth_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ckpt", default="sjtu", choices=list(CKPTS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(n_ctx=16, class_token_position="middle", csc=True, ctx_init="", prec="fp32", subsample_classes="all")
    ckpt_path, score_list = CKPTS[a.ckpt]

    pts, rgb = load_points(a.input)
    views = render_views(pts, rgb)
    imgs, deps = zip(*[to_tensor_pair(i, d) for i, d in views])
    projected_img = torch.stack(imgs).unsqueeze(0).to(device)
    depth_img = torch.stack(deps).unsqueeze(0).to(device)

    model = CLIPPCQA_Net(device, args, score_list, QUALITY_CLASSES)
    state = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path), map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model = model.to(device).eval()
    with torch.no_grad():
        _, _, quality_score, pred_dis, _ = model(projected_img, depth_img)
    dist = pred_dis[0].cpu().numpy()
    print(f"{os.path.basename(a.input)}: {len(pts)} points, ckpt={a.ckpt}, anchors={score_list}, device={device.type}")
    print("distribution over", QUALITY_CLASSES, "=", np.round(dist, 4).tolist())
    print(f"predicted score: {quality_score.item():.4f}")


if __name__ == "__main__":
    main()
