import argparse
import sys
from importlib import import_module
from pathlib import Path
import os
import wandb

os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
folder_root = Path(__file__).resolve().parents[1]
if str(folder_root) not in sys.path:
    sys.path.insert(0, str(folder_root))
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from monai.data import Dataset

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ToTensord,
)
import monai.transforms.compose as compose_mod
import monai.transforms.transform as transform_mod
import monai.utils as monai_utils
import monai.utils.misc as monai_misc
from monai.transforms.transform import Randomizable

from nucli_train.models.builders import build_model as nucli_build_model
from src.models.factory import build_local_model

FIXED_MAX = int(np.iinfo(np.uint32).max)  # 4294967295

# overwrite every cached copy of MAX_SEED
monai_utils.MAX_SEED = FIXED_MAX
monai_misc.MAX_SEED = FIXED_MAX
compose_mod.MAX_SEED = FIXED_MAX
transform_mod.MAX_SEED = FIXED_MAX

from utils.trainer_steps import train_one_epoch, validate_one_epoch


SUPPORTED_EXTS = ("*.nii.gz", "*.nii", "*.blosc2", "*.b2nd", "*.bl2")


def _strip_known_suffixes(path: Path):
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _find_pet_files(root_dir: Path, mip_subdir=None):
    files = []
    subdirs = [mip_subdir] if mip_subdir else ["*_MIPs", None]
    for subdir in subdirs:
        for ext in SUPPORTED_EXTS:
            if subdir:
                files.extend(root_dir.glob(f"*/fdg/pet/{subdir}/{ext}"))
            else:
                files.extend(root_dir.glob(f"*/fdg/pet/{ext}"))
    return sorted(set(files))


def _extract_center(file_path: Path):
    parts = file_path.parts
    if "fdg" in parts:
        fdg_idx = parts.index("fdg")
        if fdg_idx > 0:
            return parts[fdg_idx - 1]
    return file_path.parents[2].name if len(file_path.parents) > 2 else "unknown"


def _resolve_label(file_path: Path, patient_id: str, center: str, data_cfg):
    label_from = data_cfg.get("label_from", "patient_id_suffix")
    label_map = data_cfg.get("label_map", {"N": 0, "AN": 1})
    if label_from == "patient_id_suffix":
        token = patient_id.split("_")[-1]
    elif label_from == "filename_suffix":
        token = _strip_known_suffixes(file_path).split("_")[-1]
    elif label_from == "parent_dir":
        token = file_path.parent.name
    elif label_from == "center":
        token = center
    else:
        raise ValueError(f"Unsupported label_from: {label_from}")

    if token in label_map:
        return int(label_map[token])
    if str(token) in label_map:
        return int(label_map[str(token)])
    raise ValueError(
        f"Could not map label token '{token}'. "
        f"Set data.label_from/data.label_map in setup config."
    )


def create_dataframe(path_to_dataset=None, scans_excluded=None, centers_excluded=None, data_cfg=None):
    root_dir = Path(path_to_dataset)
    scans_excluded = set(scans_excluded or [])
    centers_excluded = set(centers_excluded or [])
    data_cfg = data_cfg or {}
    mip_subdir = data_cfg.get("mip_subdir")
    dataset_tag = data_cfg.get("dataset_tag", "dataset")

    save_dir = root_dir / "dataframes"
    save_dir.mkdir(parents=True, exist_ok=True)
    excluded_tag = "all" if not centers_excluded else "_".join(sorted(centers_excluded))
    save_path = save_dir / f"df_{dataset_tag}_without_{excluded_tag}.csv"

    if not save_path.exists():
        rows = []
        files = _find_pet_files(root_dir, mip_subdir=mip_subdir)
        if not files:
            raise ValueError(
                f"No files found under {root_dir}/<center>/fdg/pet/(optional *_MIPs). "
                "Set data.mip_subdir if your MIP folder name is fixed."
            )
        for file_path in files:
            center = _extract_center(file_path)
            if center in centers_excluded:
                continue
            patient_id = _strip_known_suffixes(file_path)
            if patient_id in scans_excluded or file_path.name in scans_excluded:
                continue
            label = _resolve_label(file_path, patient_id, center, data_cfg)
            rows.append(
                {
                    "PatientID": patient_id,
                    "center": center,
                    "Label": label,
                    "image": str(file_path),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No eligible files found after filtering centers/scans.")
        df.to_csv(save_path, index=False)
        print("number of patients included:", len(df))
    print("dataframe saved in:", save_path)
    return save_path


# ensure the RNG coercion can’t overflow by casting to plain Python int first
def safe_set_random_state(self, seed=None, state=None):
    if seed is not None:
        if isinstance(seed, np.generic):
            seed = seed.item()
        seed = int(seed) % FIXED_MAX
        self.R = np.random.RandomState(seed)
        return self
    if state is not None:
        if not isinstance(state, np.random.RandomState):
            raise TypeError(
                f"state must be None or np.random.RandomState, got {type(state).__name__}"
            )
        self.R = state
        return self
    self.R = np.random.RandomState()
    return self


Randomizable.set_random_state = safe_set_random_state


def build_chain(spec_list):
    return Compose([instantiate_transform(s) for s in spec_list])


def instantiate_transform(spec):
    spec = dict(spec)
    target = spec.pop("_target_")
    module_name, class_name = target.rsplit(".", 1)
    cls = getattr(import_module(module_name), class_name)
    return cls(**spec)


def _resolve_existing_path(path_value, setup_path=None):
    p = Path(str(path_value)).expanduser()
    if p.is_absolute() and p.exists():
        return p

    anchors = []
    if setup_path is not None:
        anchors.append(setup_path.parent)
    anchors.extend([Path.cwd(), repo_root, Path(__file__).resolve().parent])

    tried = []
    for base in anchors:
        candidate = (base / p).resolve()
        tried.append(candidate)
        if candidate.exists():
            return candidate

    if not p.is_absolute():
        cfg_root = repo_root / "configs"
        if cfg_root.exists():
            matches = list(cfg_root.rglob(p.name))
            if len(matches) == 1:
                return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not resolve path: {path_value}. Tried: "
        + ", ".join(str(x) for x in tried)
    )


def load_configs(args):
    setup_path = _resolve_existing_path(args.setup_config)
    with open(setup_path, "r") as f:
        setup = yaml.safe_load(f)

    transforms_path = _resolve_existing_path(
        setup["transforms_config"], setup_path=setup_path
    )
    with open(transforms_path, "r") as f:
        transforms_cfg = yaml.safe_load(f)
    return setup, transforms_cfg


def build_model(model_cfg, device):
    try:
        model = build_local_model(model_cfg)
    except ValueError:
        model = nucli_build_model(model_cfg)
    return model.to(device)


def build_optimizer_and_scheduler(model, train_cfg, epochs):
    criterion = instantiate_transform(train_cfg["criterion"])

    opt_cfg = train_cfg["optimizer"]
    lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    if opt_cfg["type"] != "adam":
        raise ValueError(f"Unsupported optimizer type: {opt_cfg['type']}")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    sched_cfg = train_cfg["scheduler"]
    if sched_cfg["type"] != "sequential":
        raise ValueError(f"Unsupported scheduler type: {sched_cfg['type']}")

    if sched_cfg["cold"]["type"] == "constant":
        cold = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=sched_cfg["cold"]["factor"],
            total_iters=sched_cfg["cold"]["total_iters"],
        )
    elif sched_cfg["cold"]["type"] == "linear":
        cold = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=sched_cfg["cold"]["factor"],
            total_iters=sched_cfg["cold"]["total_iters"],
        )
    else:
        raise ValueError(f"Unsupported cold scheduler: {sched_cfg['cold']['type']}")

    if sched_cfg["main"]["type"] == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg["main"]["step_size"],
            gamma=sched_cfg["main"]["gamma"],
        )
    elif sched_cfg["main"]["type"] == "cosineannealing":
        eta_min = float(sched_cfg["main"]["min_lr"])
        t_max = max(1, epochs - sched_cfg["cold"]["total_iters"])
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    else:
        raise ValueError(f"Unsupported main scheduler: {sched_cfg['main']['type']}")

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[cold, main],
        milestones=[sched_cfg["cold"]["total_iters"]],
    )
    return criterion, optimizer, scheduler


def main(args):
    setup, transforms_cfg = load_configs(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # INITIALIZATION
    data_cfg = setup["data"]
    model_cfg = setup["model"]
    split_cfg = setup["split"]
    train_cfg = setup["training"]
    dl_cfg = setup["dataloader"]

    path_to_dataset = data_cfg["path_to_dataset"]
    scans_excluded = data_cfg.get("scans_excluded") or []
    centers_excluded = data_cfg.get("centers_excluded")

    csv_path = create_dataframe(
        path_to_dataset=path_to_dataset,
        scans_excluded=scans_excluded,
        centers_excluded=centers_excluded,
        data_cfg=data_cfg,
    )
    label_col = "Label"
    image_col = "image"

    df = pd.read_csv(csv_path)

    records = []
    for row in df.to_dict("records"):
        patient_id = row["PatientID"]
        records.append(
            {
                "PatientID": patient_id,
                "Label": row[label_col],
                "image": row[image_col],
                "center": row.get("center", "unknown"),
            }
        )
    if not records:
        raise ValueError("No records available for training.")
    print(records[0])
    strat_labels = [f"{r['Label']}_{r['center']}" for r in records]

    cv_cfg = split_cfg.get("cross_validation", {})
    cv_enabled = bool(cv_cfg.get("enabled", False))
    n_splits = int(cv_cfg.get("n_splits", 10)) if cv_enabled else 1
    if cv_enabled and n_splits < 2:
        raise ValueError("cross_validation.n_splits must be >= 2")

    cv_fold = getattr(args, "cv_fold", None)

    if cv_enabled:
        y = strat_labels if split_cfg.get("stratify", True) else [r["Label"] for r in records]
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_cfg["seed"],
        )
        split_indices = list(splitter.split(np.arange(len(records)), y))
        if cv_fold is not None:
            if cv_fold < 1 or cv_fold > n_splits:
                raise ValueError(f"--cv-fold must be in [1, {n_splits}]")
            split_indices = [split_indices[cv_fold - 1]]
            fold_ids = [cv_fold]
        else:
            fold_ids = list(range(1, n_splits + 1))
    else:
        all_indices = np.arange(len(records))
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=split_cfg["val_ratio"],
            random_state=split_cfg["seed"],
            stratify=strat_labels if split_cfg["stratify"] else None,
        )
        split_indices = [(np.array(train_idx), np.array(val_idx))]
        fold_ids = [1]

    save_dir = Path(setup["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = save_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_dir_tail = save_dir.name
    fallback_run_name = save_dir_tail if save_dir_tail else f"{model_cfg['name']}"
    cli_run_name = getattr(args, "run_name", None)
    if cli_run_name in (None, "", "LPsmall", "give_this_a_better_name!"):
        base_run_name = fallback_run_name
    else:
        base_run_name = cli_run_name

    fold_summaries = []

    for local_idx, (train_idx, val_idx) in enumerate(split_indices):
        fold_id = fold_ids[local_idx]
        train_files = [records[i] for i in train_idx]
        val_files = [records[i] for i in val_idx]
        print(
            f"[fold {fold_id}/{n_splits if cv_enabled else 1}] "
            f"train={len(train_files)} | val={len(val_files)} | "
            f"train pos={sum(d['Label'] == 1 for d in train_files)}"
        )

        deterministic = Compose(
            [
                LoadImaged(keys=("image")),
                EnsureChannelFirstd(keys=("image")),
                EnsureTyped(keys=("image", "Label")),
            ]
        )
        random = build_chain(transforms_cfg["random"])
        val_transforms = Compose(
            [
                deterministic,
                ToTensord(keys=("image", "Label")),
            ]
        )

        train_ds = Dataset(
            data=train_files,
            transform=Compose([deterministic, random]),
        )
        val_ds = Dataset(
            data=val_files,
            transform=val_transforms,
        )

        batch_size = dl_cfg["batch_size"]
        num_workers = dl_cfg["num_workers"]
        pin_memory = dl_cfg["pin_memory"]

        sampler = None
        if dl_cfg.get("weighted_sampler", {}).get("enabled", False):
            labels = np.array([d["Label"] for d in train_files])
            class_counts = np.bincount(labels)
            weights = (1.0 / class_counts)[labels]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(weights),
                replacement=dl_cfg["weighted_sampler"].get("replacement", True),
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(
            f"[loader] batch={batch_size} | sampler={'weighted' if sampler else 'shuffle'} | workers={num_workers}"
        )

        try:
            model = build_model(model_cfg, device)
        except Exception as e:
            print("\n--- ERROR WHILE CONSTRUCTING MODEL ---")
            print("Error:", e)
            print("Type of error:", type(e))
            raise

        criterion, optimizer, scheduler = build_optimizer_and_scheduler(
            model, train_cfg, args.epochs
        )

        run_name = (
            f"{base_run_name}_fold{fold_id:02d}"
            if cv_enabled
            else base_run_name
        )
        run = wandb.init(
            project=args.experiment_name or "MAE-classifier",
            name=run_name,
            group=base_run_name if cv_enabled else None,
            reinit=True,
            config={
                "setup_config": args.setup_config,
                "nb_MIPs": data_cfg.get("nb_MIPs"),
                "batch_size": setup["dataloader"]["batch_size"],
                "lr": setup["training"]["optimizer"]["lr"],
                "model": model_cfg["name"],
                "cv_enabled": cv_enabled,
                "cv_fold": fold_id,
                "cv_n_splits": n_splits if cv_enabled else 1,
                **{
                    f"model_args.{k}": v
                    for k, v in model_cfg.get("model_args", {}).items()
                },
            },
        )

        train_loss, train_acc, train_prec, train_rec, train_auc = [], [], [], [], []
        val_loss, val_acc, val_prec, val_rec, val_auc = [], [], [], [], []

        for epoch in range(args.epochs):
            print(f"\nFold {fold_id} | Epoch {epoch+1}/{args.epochs}")
            train_metrics, _ = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_metrics, _ = validate_one_epoch(model, val_loader, criterion, device)

            train_loss.append(train_metrics.get("train_loss"))
            train_acc.append(train_metrics.get("train_acc"))
            train_prec.append(train_metrics.get("train_prec"))
            train_rec.append(train_metrics.get("train_rec"))
            train_auc.append(train_metrics.get("train_auc"))
            val_loss.append(val_metrics.get("val_loss"))
            val_acc.append(val_metrics.get("val_acc"))
            val_prec.append(val_metrics.get("val_prec"))
            val_rec.append(val_metrics.get("val_rec"))
            val_auc.append(val_metrics.get("val_auc"))

            scheduler.step()
            wandb.log(
                {
                    **train_metrics,
                    **val_metrics,
                    "fold": fold_id,
                    "epoch": epoch + 1,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/loss": train_metrics["train_loss"],
                    "val/loss": val_metrics["val_loss"],
                    "train/acc": train_metrics["train_acc"],
                    "val/acc": val_metrics["val_acc"],
                    "train/auc": train_metrics["train_auc"],
                    "val/auc": val_metrics["val_auc"],
                    "train/prec": train_metrics["train_prec"],
                    "val/prec": val_metrics["val_prec"],
                    "train/rec": train_metrics["train_rec"],
                    "val/rec": val_metrics["val_rec"],
                },
                step=epoch,
            )

        metrics_df = pd.DataFrame(
            {
                "fold": fold_id,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_prec": train_prec,
                "train_rec": train_rec,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_auc": val_auc,
            }
        )
        metrics_path = metrics_dir / f"{run_name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print("Saved metrics to", metrics_path)

        ckpt_path = save_dir / f"{run_name}.pth"
        torch.save(
            {
                "epoch": args.epochs,
                "fold": fold_id,
                "n_splits": n_splits if cv_enabled else 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_loss": train_loss[-1] if train_loss else None,
                "val_loss": val_loss[-1] if val_loss else None,
            },
            ckpt_path,
        )
        wandb.save(str(metrics_path))
        wandb.save(str(ckpt_path))
        run.finish()
        print("Saved checkpoint to", ckpt_path)

        val_auc_clean = pd.to_numeric(pd.Series(val_auc), errors="coerce")
        fold_summaries.append(
            {
                "fold": fold_id,
                "train_size": len(train_files),
                "val_size": len(val_files),
                "val_loss_last": val_loss[-1] if val_loss else None,
                "val_acc_last": val_acc[-1] if val_acc else None,
                "val_auc_last": val_auc[-1] if val_auc else None,
                "val_auc_best": (
                    float(val_auc_clean.max())
                    if not val_auc_clean.dropna().empty
                    else np.nan
                ),
            }
        )

    if fold_summaries:
        summary_df = pd.DataFrame(fold_summaries)
        summary_path = metrics_dir / (
            f"{base_run_name}_cv_summary.csv" if cv_enabled else f"{base_run_name}_summary.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        print("Saved summary to", summary_path)

        if cv_enabled and len(summary_df) > 1:
            agg_cols = ["val_loss_last", "val_acc_last", "val_auc_last", "val_auc_best"]
            means = summary_df[agg_cols].mean(numeric_only=True).to_dict()
            stds = summary_df[agg_cols].std(numeric_only=True).to_dict()
            print("[cv] means:", means)
            print("[cv] stds:", stds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-config", default="../configs/mip_setup.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--cv-fold",
        type=int,
        default=None,
        help="If set with cross_validation.enabled=true, run only one fold (1-based index).",
    )
    args = parser.parse_args()
    main(args)
