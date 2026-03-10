import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
from scipy.ndimage import map_coordinates
import argparse
from tqdm import tqdm


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
            # print("LARGE IMAGE: >630")
            start = max(0, W - target_w - 150)
            end = start + target_w
            img = img[:, start:end]
        else:
            start = max(0, W - target_w - 20)
            end = start + target_w
            img = img[:, start:end]
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
    """
    Resample a 3-D volume to an arbitrary voxel spacing.
    """
    if np.isscalar(target_mm):
        target_mm = (float(target_mm),) * 3
    if len(target_mm) != 3:
        raise ValueError("target_mm must be a float or a 3-tuple (tx, ty, tz)")

    zoom = [orig / tgt for orig, tgt in zip(voxel_mm, target_mm)]

    return ndi.zoom(vol, zoom=zoom, order=order, prefilter=prefilter, **zoom_kw)


def oblique_sagittal_mip(vol, theta_deg, order=1, cval=0.0):
    """
    Sagittal-style MIP at in-plane angle `theta_deg` without rotating
    the whole volume.  Returns an (X, Z) 2-D array.
    """
    if theta_deg == 0:
        return np.max(vol, axis=1)
    elif theta_deg == 90:
        return np.max(vol, axis=0)
    else:
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

            sampled = map_coordinates(
                vol, coords, order=order, mode="constant", cval=cval
            )
            mip[i, :] = sampled.max(axis=0)  # collapse along Y

        return mip


def main(args):

    root_dir = Path(args.root_dir)
    output_root = root_dir
    nii_paths = sorted(root_dir.glob("*.nii.gz"))
    nb_mips = 0
    for file in tqdm(nii_paths, desc="Creating MIPs"):
        center = file.parents[2].name
        tracer = file.parents[1].name
        out_dir = output_root / tracer / "pet"
        out_dir.mkdir(parents=True, exist_ok=True)

        example_output_path = out_dir / "4_MIPs" / f"4MIPs_{file.stem}.gz"
        if example_output_path.exists():
            # print("File exists already:", example_output_path)
            continue

        img = nib.load(file)
        voxel_mm = img.header.get_zooms()

        arr = img.get_fdata(dtype=np.float32)
        arr = resample_to_spacing(arr, voxel_mm, target_mm=args.target_res)
        angles = np.linspace(0, 180, args.n_angles, endpoint=False)
        mips = [oblique_sagittal_mip(arr, a) for a in angles]
        mips = [crop_pad_2d(m, args.target_hw, args.target_hw) for m in mips]
        mips = np.asarray(mips, dtype=np.float32)

        out_dir = out_dir / "4_MIPs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"4MIPs_{file.stem}.gz"
        write_nii(mips, np.eye(4), str(out_path))
        nb_mips += 1
    print("nb of mips made:", nb_mips)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--n-angles", type=int, default=4)
    parser.add_argument("--target-res", type=float, default=1.5)
    parser.add_argument("--target-hw", type=int, default=480)
    args = parser.parse_args()
    main(args)
