import argparse
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.models.MAE
from src.models.factory import build_model


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_cfg_path: str, ckpt_path: str) -> torch.nn.Module:
    cfg = yaml.safe_load(open(model_cfg_path))
    model_cfg = cfg["model"] if "model" in cfg else cfg

    model = build_model(model_cfg)

    moved = model.cuda()
    if moved is not None:
        model = moved
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cuda")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    remapped = {}
    for k, v in state.items():
        if k.startswith("network."):
            remapped["network." + k] = v
        else:
            remapped["network.network." + k] = v

    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in remapped.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded checkpoint: matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    print("missing checkpoints:", missing)

    # User asked to apply the validation log1p preprocessing.
    model.intensitylog = True
    return model


def load_mips_nii(nii_path: str, expected_mips: int = 4) -> tuple[torch.Tensor, str]:
    p = Path(nii_path)
    vol = nib.load(str(p)).get_fdata().astype(np.float32)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D MIP volume, got {vol.shape}")

    if vol.shape[-1] == expected_mips:
        mips = np.moveaxis(vol, -1, 0)
    elif vol.shape[0] == expected_mips:
        mips = vol
    else:
        raise ValueError(f"Expected {expected_mips} MIPs, got shape {vol.shape}")

    x = torch.from_numpy(mips).unsqueeze(1)
    return x, p.stem


def to_display(t: torch.Tensor, intensitylog: bool) -> torch.Tensor:
    if intensitylog:
        return torch.expm1(t)
    return t


def to_plot_image(chw: torch.Tensor, rotate_k: int):
    arr = chw.detach().cpu().numpy()
    c = arr.shape[0]
    if c > 1:
        img = np.rot90(arr.mean(axis=0), k=rotate_k)
        return img, {"cmap": "gray"}
    img = np.rot90(arr[0], k=rotate_k)
    return img, {"cmap": "gray"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg", required=True, help="Path to MAE model cfg YAML")
    parser.add_argument("--ckpt-path", required=True, help="Path to MAE checkpoint")
    parser.add_argument(
        "--mip-nii", required=True, help="Path to one .nii.gz containing 4 MIPs"
    )
    parser.add_argument(
        "--out-path", default=None, help="Output PNG path (default: next to input)"
    )
    parser.add_argument(
        "--rotate-k", type=int, default=1, help="k for np.rot90 display orientation"
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=10.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for validation mask sampling.",
    )
    args = parser.parse_args()

    print("using ckpt path:", args.ckpt_path)
    print("using mip:", args.mip_nii)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required because MAE.validation_step uses .cuda() internally."
        )

    set_random_seed(args.seed)
    print(f"using seed: {args.seed}")
    model = load_model(args.model_cfg, args.ckpt_path)
    x_raw, patient_id = load_mips_nii(args.mip_nii, expected_mips=4)

    with torch.no_grad():
        out = model.validation_step(
            {"data": x_raw, "patient_id": [patient_id] * x_raw.shape[0]}
        )

    intensitylog = bool(out.get("intensitylog", False))
    mask = out["mask"].detach().cpu()
    inp = to_display(out["input"].detach().cpu(), intensitylog)
    preds_disp = to_display(out["predictions"].detach().cpu(), intensitylog)
    if mask.shape[1] == 1 and inp.shape[1] > 1:
        mask = mask.expand(-1, inp.shape[1], -1, -1)
    recon = inp * mask + preds_disp * (1 - mask)

    masked = to_display(out["masked_data"].detach().cpu(), intensitylog)

    n = inp.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i in range(n):
        panels = [
            (inp[i], f"Original (MIP {i})"),
            (masked[i], "Masked"),
            (recon[i], "Reconstruction"),
        ]
        for j, (img, title) in enumerate(panels):
            plot_img, plot_kwargs = to_plot_image(img, args.rotate_k)
            axes[i, j].imshow(
                plot_img,
                cmap=plot_kwargs["cmap"],
                vmin=args.vmin,
                vmax=args.vmax,
            )
            axes[i, j].set_title(title)
            axes[i, j].axis("off")

    out_path = (
        Path(args.out_path)
        if args.out_path
        else Path(args.mip_nii).with_suffix("").with_suffix("").parent
        / f"{patient_id}_mae_validation_triptych.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
