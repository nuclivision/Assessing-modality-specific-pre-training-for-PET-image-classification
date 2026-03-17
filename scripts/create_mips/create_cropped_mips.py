import argparse
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates
from tqdm import tqdm


def crop_pad_2d(img, target_h, target_w):
    """Center-crop or zero-pad a 2-D array to (target_h, target_w)."""
    H, W = img.shape

    # Height
    dh = target_h - H
    if dh >= 0:
        top = dh // 2
        img = np.pad(img, ((top, dh - top), (0, 0)), mode="constant")
    else:
        cut = -dh
        top = cut // 2
        img = img[top : H - (cut - top), :]

    # Width
    dw = target_w - W
    if dw >= 0:
        left = dw // 2
        img = np.pad(img, ((0, 0), (left, dw - left)), mode="constant")
    else:
        cut = -dw
        left = cut // 2
        img = img[:, left : W - (cut - left)]

    return img


def write_nii(array, affine_ras, full_filename_nii):
    nii_img = nib.Nifti1Image(array, affine_ras)
    nii_img.set_sform(affine_ras, code=1)
    nib.save(nii_img, full_filename_nii)


def resample_to_spacing(
    vol: np.ndarray,
    voxel_mm: tuple,
    target_mm=2.0,
    order: int = 1,
    prefilter: bool = False,
    **zoom_kw,
):
    """Resample a 3-D volume to isotropic or anisotropic voxel spacing."""
    if np.isscalar(target_mm):
        target_mm = (float(target_mm),) * 3
    if len(target_mm) != 3:
        raise ValueError("target_mm must be a float or a 3-tuple (tx, ty, tz)")
    zoom = [orig / tgt for orig, tgt in zip(voxel_mm, target_mm)]
    return ndi.zoom(vol, zoom=zoom, order=order, prefilter=prefilter, **zoom_kw)


def oblique_sagittal_mip(vol, theta_deg, order=1, cval=0.0):
    """
    Sagittal-style MIP at in-plane angle `theta_deg` without rotating
    the whole volume. Expects vol in (X, Y, Z) order with Z as the axial
    (head-to-foot) dimension. Returns a 2-D array of shape (X, Z).
    """
    if vol.shape[0] >= vol.shape[2]:
        warnings.warn(
            f"Expected axis 0 (X) to be shorter than axis 2 (Z/axial), "
            f"got shape {vol.shape}. Transposing to (Z, Y, X) — verify your "
            f"volume orientation.",
            stacklevel=2,
        )
        vol = np.transpose(vol, (2, 1, 0))

    if theta_deg == 0:
        return np.max(vol, axis=1)
    if theta_deg == 90:
        return np.max(vol, axis=0)

    X, Y, Z = vol.shape
    cx, cy = (X - 1) / 2.0, (Y - 1) / 2.0

    th = np.deg2rad(theta_deg)
    cth, sth = np.cos(th), np.sin(th)

    y_idx = np.arange(Y)[:, None] - cy  # (Y, 1)
    z_idx = np.arange(Z)[None, :]  # (1, Z)

    mip = np.full((X, Z), cval, dtype=vol.dtype)
    for i, dx in enumerate(np.arange(X) - cx):
        xs = cx + cth * dx + sth * y_idx  # (Y, 1)
        ys = cy - sth * dx + cth * y_idx  # (Y, 1)

        xs = np.repeat(xs, Z, axis=1)  # (Y, Z)
        ys = np.repeat(ys, Z, axis=1)  # (Y, Z)
        zs = np.repeat(z_idx, Y, axis=0)  # (Y, Z)

        sampled = map_coordinates(
            vol, np.stack([xs, ys, zs], axis=0), order=order, mode="constant", cval=cval
        )
        mip[i, :] = sampled.max(axis=0)

    return mip


def main(args):
    root_dir = Path(args.root_dir)
    nii_paths = sorted(root_dir.glob("*/*.nii.gz"))

    if not nii_paths:
        raise FileNotFoundError(f"No .nii.gz files found under {root_dir}/*/*.nii.gz")

    print(f"Found {len(nii_paths)} scan(s) — creating {args.n_angles}-angle MIPs.")
    nb_created = 0
    nb_skipped = 0

    for file in tqdm(nii_paths, desc="Creating MIPs"):
        tracer = file.parents[1].name
        out_dir = root_dir / tracer / "pet" / f"{args.n_angles}_MIPs"
        out_path = out_dir / f"{args.n_angles}MIPs_{file.stem}.gz"

        if out_path.exists():
            nb_skipped += 1
            continue

        img = nib.load(file)
        arr = img.get_fdata(dtype=np.float32)
        arr = resample_to_spacing(
            arr, img.header.get_zooms(), target_mm=args.target_res
        )

        angles = np.linspace(0, 180, args.n_angles, endpoint=False)
        mips = [oblique_sagittal_mip(arr, a) for a in angles]
        mips = [crop_pad_2d(m, args.target_hw, args.target_hw) for m in mips]
        mips = np.asarray(mips, dtype=np.float32)

        out_dir.mkdir(parents=True, exist_ok=True)
        write_nii(mips, np.eye(4), str(out_path))
        nb_created += 1

    print(f"Done. Created: {nb_created} | Skipped (already exist): {nb_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create oblique sagittal MIPs from PET NIfTI files."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing <tracer>/<scan>.nii.gz files.",
    )
    parser.add_argument(
        "--n-angles",
        type=int,
        default=4,
        help="Number of MIP angles evenly spaced over [0, 180) degrees.",
    )
    parser.add_argument(
        "--target-res",
        type=float,
        default=1.5,
        help="Target isotropic voxel spacing in mm before MIP projection.",
    )
    parser.add_argument(
        "--target-hw",
        type=int,
        default=480,
        help="Target height and width of the output MIP images in pixels.",
    )
    args = parser.parse_args()
    main(args)
