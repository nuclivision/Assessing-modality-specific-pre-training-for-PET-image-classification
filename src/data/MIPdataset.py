import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.val.evaluator_MAE import MAEevaluator


SUPPORTED_EXTS = ("*.nii.gz", "*.nii")


def _list_volume_files(root_dir: Path, pattern_prefix: str) -> list:
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(root_dir.glob(f"{pattern_prefix}/{ext}"))
    return sorted(files)


class MIPDataset(Dataset):
    """
    Dataset that loads 2-D MIP slices from 3-D NIfTI volumes.

    Each volume contributes one entry per slice along `slice_axis`.
    Optionally extracts a random square patch, retrying until the patch
    contains enough foreground (controlled by `min_foreground_ratio`).
    """

    def __init__(
        self,
        root_dir,
        records,
        transforms=None,
        patch_size=None,
        slice_axis=0,
        min_foreground_ratio=0.0,
        max_slices_per_volume=None,
    ):
        self.root = Path(root_dir)
        self.records = records
        self.transforms = transforms
        self.slice_axis = slice_axis
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.max_slices_per_volume = max_slices_per_volume

        if patch_size is not None:
            self.dx, self.dy = patch_size[0] // 2, patch_size[1] // 2
        else:
            self.dx = self.dy = None

        self.mip_entries = []
        for rec in records:
            vol_path = self._resolve_path(rec["image"])
            n_slices = nib.load(vol_path).shape[slice_axis]
            if max_slices_per_volume is not None:
                n_slices = min(n_slices, max_slices_per_volume)
            for mip_idx in range(n_slices):
                self.mip_entries.append((rec, mip_idx))

    def _resolve_path(self, path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def __len__(self):
        return len(self.mip_entries)

    def __getitem__(self, idx):
        rec, mip_idx = self.mip_entries[idx]
        vol = np.asarray(
            nib.load(self._resolve_path(rec["image"])).dataobj, dtype=np.float32
        )

        if vol.ndim == 2:
            arr = vol
        elif vol.ndim == 3:
            arr = np.take(vol, mip_idx, axis=self.slice_axis)
        else:
            raise ValueError(f"Unsupported volume shape {vol.shape} for {rec['image']}")

        mip = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)

        if self.transforms:
            mip = self.transforms(mip)

        if self.dx is not None:
            max_tries = 30
            for _ in range(max_tries):
                patch = self._random_patch(mip[0])
                if (patch > 0).float().mean().item() >= self.min_foreground_ratio:
                    break
            mip = patch.unsqueeze(0)

        return {"data": mip, "patient_id": rec["PatientID"]}

    def _random_patch(self, img2d) -> torch.Tensor:
        """Sample a random (dx*2, dy*2) patch from a 2-D tensor."""
        H, W = img2d.shape
        coords = []
        pad = []
        for size, half in zip((H, W), (self.dx, self.dy)):
            if size < 2 * half:
                coords.append((0, size))
                pad.append(2 * half - size)
            else:
                c = random.randint(half, size - half)
                coords.append((c - half, c + half))
                pad.append(0)

        (r0, r1), (c0, c1) = coords
        patch = img2d[r0:r1, c0:c1]

        if any(p > 0 for p in pad):
            patch = torch.nn.functional.pad(
                patch.unsqueeze(0).unsqueeze(0),
                (0, pad[1], 0, pad[0]),
            ).squeeze()

        return patch


def mip_augmentation(mip, flipped=True, rotated=True):
    """Random horizontal flip and 90-degree rotation for a MIP tensor."""
    if flipped and torch.rand(1).item() < 0.5:
        mip = torch.flip(mip, dims=[-1])
    if rotated:
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            mip = torch.rot90(mip, k=k, dims=[1, 2])
    return mip


def create_dataframe(
    path_to_dataset,
    scans_excluded=None,
    nb_MIPs=None,
    centers_excluded=None,
    source_subdir=None,
):
    """
    Scan the dataset directory and build a patient CSV, cached to disk.

    Expected structure: ``<root>/<center>/fdg/pet/<source_subdir>/<file>.nii.gz``

    Args:
        path_to_dataset: root directory of the dataset.
        scans_excluded:  list of PatientIDs to skip.
        nb_MIPs:         number of MIPs per volume (inferred from data if None).
        centers_excluded: list of center names to skip.
        source_subdir:   subdirectory inside ``pet/`` to search (e.g. ``"4_MIPs"``).
    """
    root_dir = Path(path_to_dataset)
    scans_excluded = set(scans_excluded or [])
    centers_excluded = set(centers_excluded or [])

    save_dir = root_dir / "dataframes"
    save_dir.mkdir(parents=True, exist_ok=True)

    if centers_excluded:
        excluded_tag = "_".join(sorted(centers_excluded))
        save_path = save_dir / f"df_mips_without_{excluded_tag}.csv"
    else:
        save_path = save_dir / "df_mips.csv"

    if save_path.exists():
        existing = pd.read_csv(save_path)
        if len(existing) > 0:
            print(f"Using existing dataframe ({len(existing)} patients): {save_path}")
            return str(save_path)

    subdir = source_subdir if source_subdir is not None else "4_MIPs"
    files = _list_volume_files(root_dir, f"*/fdg/pet/{subdir}")

    if not files:
        raise ValueError(
            f"No NIfTI files found under {root_dir}/*/fdg/pet/{subdir}/. "
            "Check path_to_dataset and source_subdir."
        )

    if nb_MIPs is None:
        nb_MIPs = nib.load(files[0]).shape[0]

    rows = []
    for file in files:
        center = file.parents[2].name
        if center in centers_excluded:
            continue
        pid = file.name.split(".")[0]
        if pid in scans_excluded:
            continue
        rows.append(
            {
                "PatientID": pid,
                "center": center,
                "image": str(file),
                "nb_MIPs": nb_MIPs,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No eligible files found after filtering. "
            f"data_root={path_to_dataset}, source_subdir={source_subdir}."
        )
    df.to_csv(save_path, index=False)
    print(f"Dataset saved ({len(df)} patients): {save_path}")
    return str(save_path)


def build_mip_splits(
    path_to_dataset, nb_MIPs, split_cfg, scans_excluded=None, source_subdir=None
):
    """Load the dataset CSV and split into train/val record lists."""
    csv_path = create_dataframe(
        path_to_dataset, scans_excluded, nb_MIPs, source_subdir=source_subdir
    )
    df = pd.read_csv(csv_path)
    records = [
        {"PatientID": row["PatientID"], "image": row["image"], "center": row["center"]}
        for row in df.to_dict("records")
    ]
    print(f"Total records: {len(records)}")

    val_ratio = split_cfg.get("val_ratio")
    if not val_ratio:
        return records, []

    train_records, val_records = train_test_split(
        records,
        test_size=val_ratio,
        random_state=split_cfg["seed"],
        stratify=[r["center"] for r in records] if split_cfg.get("stratify") else None,
    )
    print(f"Train: {len(train_records)} | Val: {len(val_records)}")
    return train_records, val_records


def build_transforms(transforms: list):
    flipped = "flip" in transforms
    rotated = "rotate" in transforms
    return lambda x: mip_augmentation(x, flipped=flipped, rotated=rotated)


def build_mip_data(cfg):
    """Build train DataLoader config and val DataLoaders from a data config dict or YAML path."""
    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = yaml.safe_load(f)

    train_records, val_records = build_mip_splits(
        path_to_dataset=cfg["data_root"],
        nb_MIPs=cfg.get("nb_MIPs"),
        split_cfg=cfg["split"],
        scans_excluded=cfg.get("scans_excluded", []),
        source_subdir=cfg.get("source_subdir"),
    )

    dataset_kwargs = dict(
        patch_size=cfg.get("patch_size"),
        slice_axis=cfg.get("slice_axis", 0),
        min_foreground_ratio=cfg.get("min_foreground_ratio", 0.0),
        max_slices_per_volume=cfg.get("max_slices_per_volume"),
    )
    transforms = cfg.get("transforms", [])
    train_transform = build_transforms(transforms) if transforms else None

    train_dataset = MIPDataset(
        cfg["data_root"], train_records, transforms=train_transform, **dataset_kwargs
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty. Check data_root, source_subdir, and scan filters."
        )

    val_dataset = MIPDataset(cfg["data_root"], val_records, **dataset_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["val"]["batch_size"],
        shuffle=False,
        num_workers=cfg["val"]["num_workers"],
        drop_last=True,
    )

    print(
        f"MIPs in train: {len(train_dataset.mip_entries)} | val: {len(val_dataset.mip_entries)}"
    )

    return (
        {
            "dataset": train_dataset,
            "batch_size": cfg["train"]["batch_size"],
            "num_workers": cfg["train"]["num_workers"],
        },
        {
            "mip_val": {
                "interval": cfg["val"].get("global_eval_interval", 1),
                "loader": val_loader,
                "evaluators": [MAEevaluator("examples/mips")],
            }
        },
    )
