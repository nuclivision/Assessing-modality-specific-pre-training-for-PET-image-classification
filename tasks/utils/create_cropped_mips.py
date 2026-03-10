import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
from scipy.ndimage import map_coordinates
import argparse


def crop_pad_2d(img, target_h, target_w):
    H, W = img.shape
    dh = target_h - H
    dw = target_w - W

    if dh >= 0:
        top = dh // 2
        bottom = dh - top
        img = np.pad(img, ((top, bottom), (0, 0)), mode="constant")
    else:
        cut = -dh
        top = cut // 2
        bottom = cut - top
        img = img[top : H - bottom, :]

    if dw >= 0:
        left = dw // 2
        right = dw - left
        img = np.pad(img, ((0, 0), (left, right)), mode="constant")
    else:
        if W > 600:
            print("LARGE IMAGE: >630")
            start = max(0, W - target_w - 150)
            end = start + target_w
            img = img[:, start:end]
        else:
            start = max(0, W - target_w - 20)
            end = start + target_w
            img = img[:, start:end]
    return img


def write_nii(array, affine_ras, full_filename_nii):
    """
    Convert a 3D array with an affine matrix to a NIfTI file.
    Applies a flip matrix to convert from LPS (NRRD convention) to RAS (NIfTI convention).
    """

    # Create the NIfTI image
    nii_img = nib.Nifti1Image(array, affine_ras)

    # Explicitly set the sform and sform code
    nii_img.set_sform(affine_ras, code=1)

    # Save the NIfTI file
    nib.save(nii_img, full_filename_nii)


def resample_to_spacing(
    vol: np.ndarray,
    voxel_mm: tuple,
    target_mm=2.0,
    order: int = 1,
    prefilter: bool = False,
    **zoom_kw,
):
    """
    Resample a 3-D volume to an arbitrary voxel spacing.
    """
    if np.isscalar(target_mm):
        target_mm = (float(target_mm),) * 3
    if len(target_mm) != 3:
        raise ValueError("target_mm must be a float or a 3-tuple (tx, ty, tz)")

    print(f"target mm: {target_mm}")
    zoom = [orig / tgt for orig, tgt in zip(voxel_mm, target_mm)]

    return ndi.zoom(vol, zoom=zoom, order=order, prefilter=prefilter, **zoom_kw)


def oblique_sagittal_mip(vol, theta_deg, order=1, cval=0.0):
    """
    Sagittal-style MIP at in-plane angle `theta_deg` without rotating
    the whole volume.  Returns an (X, Z) 2-D array.
    """
    X, Y, Z = vol.shape
    cx, cy = (X - 1) / 2.0, (Y - 1) / 2.0

    th = np.deg2rad(theta_deg)
    cth, sth = np.cos(th), np.sin(th)

    # Pre-compute indices that do not depend on column i
    y_idx = np.arange(Y)[:, None] - cy  # shape (Y,1)
    z_idx = np.arange(Z)[None, :]  # shape (1,Z), later broadcast

    mip = np.full((X, Z), cval, dtype=vol.dtype)  # output buffer

    x_idx = np.arange(X) - cx  # centred column offsets
    for i, dx in enumerate(x_idx):
        # Parametric line for this column → arrays of shape (Y,1)
        xs = cx + cth * dx + sth * y_idx
        ys = cy - sth * dx + cth * y_idx

        # ---- tile to (Y,Z) so shapes match ---------------------------
        xs = np.repeat(xs, Z, axis=1)  # (Y,Z)
        ys = np.repeat(ys, Z, axis=1)  # (Y,Z)
        zs = np.repeat(z_idx, Y, axis=0)  # (Y,Z)

        coords = np.stack([xs, ys, zs], axis=0)  # (3, Y, Z)

        sampled = map_coordinates(vol, coords, order=order, mode="constant", cval=cval)
        mip[i, :] = sampled.max(axis=0)  # collapse along Y

    print(f"Initial volume shape: {vol.shape}, end MIP shape: {mip.shape}")

    return mip


def main(args):

    N_ANGLES = 4
    TARGET_RES = 1.5
    TARGET_MIP_H = 480
    TARGET_MIP_W = 480

    root_dir = Path(args.root_dir)
    output_root = root_dir
    nii_paths = sorted(root_dir.glob("*.nii.gz"))
    nb_mips = 0
    for file in nii_paths:
        center = file.parents[2].name
        tracer = file.parents[1].name
        out_dir = output_root / tracer / "pet"
        out_dir.mkdir(parents=True, exist_ok=True)

        example_output_path = out_dir / "4_MIPs" / f"4MIPs_{file.stem}.gz"
        if example_output_path.exists():
            print("File exists already:", example_output_path)
            continue

        img = nib.load(file)
        voxel_mm = img.header.get_zooms()

        arr = img.get_fdata()
        arr = resample_to_spacing(arr, voxel_mm, target_mm=TARGET_RES)
        angles = np.linspace(0, 180, N_ANGLES, endpoint=False)
        mips = [oblique_sagittal_mip(arr, a) for a in angles]
        mips = [crop_pad_2d(m, TARGET_MIP_H, TARGET_MIP_W) for m in mips]
        mips = np.asarray(mips, dtype=np.float32)

        out_dir = out_dir / "4_MIPs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"4MIPs_{file.stem}.gz"
        write_nii(mips, np.eye(4), str(out_path))
        nb_mips += 1
    print("nb of mips made:", nb_mips)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="/mnt/c/Users/roxan/OS_data/")
    args = parser.parse_args()
    main(args)
