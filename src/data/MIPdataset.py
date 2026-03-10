import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import numpy as np
import random


from src.val.evaluator_MAE import MAEevaluator


def _open_volume(path: Path):
    return "nifti", nib.load(path)


def _list_volume_files(root_dir, pattern_prefix):
    files = []
    for ext in ("*.nii.gz", "*.nii", "*.blosc2", "*.b2nd", "*.bl2"):
        files.extend(root_dir.glob(f"{pattern_prefix}/{ext}"))
    return sorted(files)


class MIPDataset(Dataset):
    def __init__(
        self,
        root_dir,
        records,
        transforms=None,
        patch_size=[80, 80],
        slice_axis=0,
        min_foreground_ratio=0.0,
        max_slices_per_volume=None,
    ):
        self.root = Path(root_dir)
        self.records = records
        self.transforms = transforms
        self.mip_entries = []
        self.dx = None
        if patch_size is not None:
            self.dx, self.dy = list((np.array(patch_size) // 2))
        self.slice_axis = slice_axis
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.max_slices_per_volume = max_slices_per_volume
        for rec in records:
            vol_path = Path(rec["image"])
            if not vol_path.is_absolute():
                vol_path = self.root / vol_path
            backend, vol_img = _open_volume(vol_path)
            vol_shape = vol_img.shape
            vol_ndim = len(vol_shape)

            num_entries = vol_shape[self.slice_axis] if vol_ndim == 3 else 1
            indices = list(range(num_entries))
            for mip_idx in indices:
                self.mip_entries.append((rec, int(mip_idx)))

    def __len__(self):
        return len(self.mip_entries)

    def __getitem__(self, idx):
        rec, mip_idx = self.mip_entries[idx]
        vol_path = Path(rec["image"])
        if not vol_path.is_absolute():
            vol_path = self.root / vol_path
        backend, vol_obj = _open_volume(vol_path)
        vol_shape = vol_obj.shape
        vol_ndim = len(vol_shape)

        if vol_ndim == 2:
            if backend == "nifti":
                arr = np.asarray(vol_obj.dataobj, dtype=np.float32)
            else:
                arr = np.asarray(vol_obj[:], dtype=np.float32)
            arr = np.array(arr, dtype=np.float32, copy=True)
            mip = torch.from_numpy(arr).float().unsqueeze(0)
        elif vol_ndim == 3:
            if backend == "nifti":
                arr3d = np.asarray(vol_obj.dataobj, dtype=np.float32)
            else:
                arr3d = np.asarray(vol_obj[:], dtype=np.float32)
            arr3d = np.array(arr3d, dtype=np.float32, copy=True)

            if self.slice_axis == 0:
                arr = arr3d[mip_idx, :, :]
            elif self.slice_axis == 1:
                arr = arr3d[:, mip_idx, :]
            elif self.slice_axis == 2:
                arr = arr3d[:, :, mip_idx]
            else:
                raise ValueError(
                    f"Unsupported slice_axis={self.slice_axis}. Expected 0, 1 or 2."
                )

            mip = torch.from_numpy(arr).float().unsqueeze(0)
        else:
            raise ValueError(f"Unsupported shape {vol_shape} for record {rec['image']}")
        C, H, W = mip.shape
        if self.transforms:
            mip = self.transforms(mip)
        # Select a patch
        if self.dx:
            max_tries = 30
            patch = None
            for mip2d in mip:
                for _ in range(max_tries):
                    patch = self.load_patch(mip2d)
                    if torch.is_tensor(patch):
                        black = (patch <= 0).float().mean().item()
                    else:
                        black = float(np.mean(np.asarray(patch) <= 0))
                    if black <= 0.5:
                        break
            mip = patch.unsqueeze(0)
        sample = {"data": mip, "patient_id": rec["PatientID"]}
        return sample

    def load_patch(self, mip, coords=None):
        if coords is not None:
            s = mip.shape
            pad_list = []
            scope = []

            for i, (dm, m_c) in enumerate(zip([self.dx, self.dy], coords)):
                scope.append(
                    (
                        m_c - dm if m_c - dm >= 0 else None,
                        dm + m_c if dm + m_c < s[i] else None,
                    )
                )
                pad_list.append(
                    (
                        0 if m_c - dm >= 0 else abs(dm - m_c),
                        0 if dm + m_c <= s[i] else dm + m_c - s[i],
                    )
                )
            x, y = scope
            patch = mip[x[0] : x[1], y[0] : y[1]]
            return np.pad(patch, pad_list)
        else:
            s = mip.shape
            self.coords = []
            self.pad_list = []
            for i, dm in enumerate([self.dx, self.dy]):
                if s[i] - 2 * dm < 0:
                    self.coords.append((None, None))
                    self.pad_list.append((2 * dm - s[i], 0))
                else:
                    m_c = random.randint(dm, s[i] - dm)
                    self.coords.append((m_c - dm, m_c + dm))
                    self.pad_list.append((0, 0))
            x, y = self.coords
            non_full_patch = mip[x[0] : x[1], y[0] : y[1]]
            patch = np.pad(non_full_patch, self.pad_list)
            return torch.from_numpy(patch).float()


def mip_augmentation(mip, flipped=True, rotated=True):

    if flipped:
        if torch.rand(1).item() < 0.5:
            mip = torch.flip(mip, dims=[-1])

    if rotated:
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            mip = torch.rot90(mip, k=k, dims=[1, 2])

    return mip


def create_dataframe(
    path_to_dataset=None,
    scans_excluded=None,
    nb_MIPs=None,
    centers_excluded=None,
    source_subdir=None,
):
    """
    Creates dataframe for general pre-training data
    does not need a label
    only required params are path_to_dataset and nb_MIPs
    path_to_dataset: str of the path to the dataset, in which files are stored like path -> center -> fdg -> pet -> {nb_MIPs}_MIPs -> .nii.gz
    nb_MIPs: the number of MIPs you divided your data in
    """
    root_dir = Path(path_to_dataset)
    scans_excluded = scans_excluded or []
    save_dir = path_to_dataset + "dataframes"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if centers_excluded == None or centers_excluded == []:
        save_path = save_dir + f"/df_mips.csv"
    else:
        save_path = save_dir + f"/df_mips_without_{centers_excluded[:]}.csv"
    existing_ok = False
    if Path(save_path).exists():
        try:
            existing_df = pd.read_csv(save_path)
            existing_ok = len(existing_df) > 0
        except Exception:
            existing_ok = False

    if not existing_ok:
        rows = []
        if source_subdir is not None:
            glob_root = _list_volume_files(root_dir, f"*/fdg/pet/{source_subdir}")
        else:
            glob_root = _list_volume_files(root_dir, f"*/fdg/pet/4_MIPs")
        if nb_MIPs is None:
            candidates = _list_volume_files(root_dir, "*/fdg/pet")
            sample = next(iter(candidates), None)
            if sample is None:
                raise ValueError(
                    f"No volume file found under {root_dir}/*/fdg/pet to infer nb_MIPs."
                )
            _, sample_vol = _open_volume(sample)
            nb_MIPs = sample_vol.shape[0]
            glob_root = candidates

        for file in glob_root:
            center = file.parents[2].name
            pid = file.name.split(".")[0]
            if centers_excluded is not None:
                if center in centers_excluded:
                    continue
            if pid not in scans_excluded:
                rows.append(
                    {
                        "PatientID": pid,
                        "center": center,
                        "image": file,
                        "nb_MIPs": nb_MIPs,
                    }
                )
        df = pd.DataFrame(rows)

        df.to_csv(save_path)
        print("number of patients included:", len(df))
        if len(df) == 0:
            raise ValueError(
                "No files found for dataframe creation. "
                f"data_root={path_to_dataset}, source_subdir={source_subdir}. "
                "Expected 3D volumes in */fdg/pet/*.nii.gz (or *.nii or *.blosc2)."
            )
    print("dataframe saved in:", save_path)

    return save_path


def build_mip_splits(
    path_to_dataset,
    nb_MIPs,
    split_cfg,
    scans_excluded=None,
    source_subdir=None,
):

    csv_path = create_dataframe(
        path_to_dataset,
        scans_excluded,
        nb_MIPs,
        None,
        source_subdir=source_subdir,
    )
    df = pd.read_csv(csv_path)
    print("length of data:", len(df))
    records = []
    for row in df.to_dict("records"):
        patient_id = row["PatientID"]
        center = row["center"]
        records.append(
            {
                "PatientID": patient_id,
                "image": row["image"],  # relative path to NIfTI
                "center": center,
            }
        )

    strat = [f"{r['center']}" for r in records]
    val_ratio = split_cfg.get("val_ratio")
    if not val_ratio:
        return records, []
    train_records, val_records = train_test_split(
        records,
        test_size=val_ratio,
        random_state=split_cfg["seed"],
        stratify=strat if split_cfg.get("stratify") else None,
    )
    print(
        f"length of training data: {len(train_records)}, length of val data: {len(val_records)}"
    )
    return train_records, val_records


def build_transforms(transforms):
    flipped = "flip" in transforms
    rotated = "rotate" in transforms
    return lambda x: mip_augmentation(x, flipped=flipped, rotated=rotated)


def build_mip_data(cfg):
    # cfg can be a dict or a yaml path
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)

    path_to_dataset = cfg["data_root"]
    nb_MIPs = cfg.get("nb_MIPs")
    split_cfg = cfg["split"]
    centers = cfg.get("centers")
    centers_excluded = cfg.get("centers_excluded")
    scans_excluded = cfg.get("scans_excluded", [])
    transforming = cfg.get("transforms", [])
    patch_size = cfg.get("patch_size", None)
    source_subdir = cfg.get("source_subdir")
    slice_axis = cfg.get("slice_axis", 0)
    min_foreground_ratio = cfg.get("min_foreground_ratio", 0.0)
    max_slices_per_volume = cfg.get("max_slices_per_volume")
    train_transform = build_transforms(transforming) if transforming else None
    train_records, val_records = build_mip_splits(
        path_to_dataset=path_to_dataset,
        nb_MIPs=nb_MIPs,
        split_cfg=split_cfg,
        scans_excluded=scans_excluded,
        source_subdir=source_subdir,
    )
    train_dataset = MIPDataset(
        path_to_dataset,
        train_records,
        patch_size=patch_size,
        transforms=train_transform,
        slice_axis=slice_axis,
        min_foreground_ratio=min_foreground_ratio,
        max_slices_per_volume=max_slices_per_volume,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty. Check data_root/sample_mode/source_subdir and scan filters."
        )
    train_data = {
        "dataset": train_dataset,
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": cfg["train"]["num_workers"],
    }

    val_dataset = MIPDataset(
        path_to_dataset,
        val_records,
        patch_size=patch_size,
        slice_axis=slice_axis,
        min_foreground_ratio=min_foreground_ratio,
        max_slices_per_volume=max_slices_per_volume,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["val"]["batch_size"],
        shuffle=False,
        num_workers=cfg["val"]["num_workers"],
        drop_last=True,
    )
    print(f"Total nb of MIPs in train: {len(train_dataset.mip_entries)}")
    print(f"Total nb of MIPs in val: {len(val_dataset.mip_entries)}")
    evaluators = [MAEevaluator(f"examples/mips")]
    val_loaders = {
        "mip_val": {
            "interval": cfg["val"].get("global_eval_interval", 1),
            "loader": val_loader,
            "evaluators": evaluators,
        }
    }
    return train_data, val_loaders
