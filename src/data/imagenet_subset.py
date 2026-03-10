import random
from pathlib import Path

import yaml
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.val.evaluator_MAE_imagenet import MAEevaluatorIN


class ImageFolderAsDict(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        return {"data": image, "label": target}


class ShiftMinToZero:
    def __init__(self, per_channel=True, fixed_shift=None):
        self.per_channel = per_channel
        self.fixed_shift = fixed_shift

    def __call__(self, x):
        if self.fixed_shift is not None:
            return x + float(self.fixed_shift)
        if self.per_channel:
            return x - x.amin(dim=(1, 2), keepdim=True)
        return x - x.amin()


def _get_mean_std(cfg):
    if cfg.get("imagenet_default_mean_and_std", True):
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def _resolve_class_ids(base_dataset, cfg, seed):
    subset_classes = cfg.get("subset_classes")
    subset_num_classes = cfg.get("subset_num_classes")
    class_seed = cfg.get("subset_class_seed", seed)
    if subset_classes:
        return [base_dataset.class_to_idx[c] for c in subset_classes]
    if subset_num_classes:
        rng = random.Random(class_seed)
        all_classes = list(base_dataset.class_to_idx.values())
        subset_num_classes = min(subset_num_classes, len(all_classes))
        return rng.sample(all_classes, subset_num_classes)
    return None


def _select_subset_indices(base_dataset, subset_size, seed, per_class, class_ids=None):
    num_samples = len(base_dataset)
    if subset_size is None or subset_size <= 0 or subset_size >= num_samples:
        if class_ids is None:
            return list(range(num_samples))
        return [
            i for i, target in enumerate(base_dataset.targets) if target in class_ids
        ]

    rng = random.Random(seed)
    if not per_class:
        indices = list(range(num_samples))
        if class_ids is not None:
            indices = [
                i
                for i, target in enumerate(base_dataset.targets)
                if target in class_ids
            ]
        if subset_size >= len(indices):
            return indices
        return rng.sample(indices, subset_size)

    class_to_indices = {}
    for idx, target in enumerate(base_dataset.targets):
        if class_ids is not None and target not in class_ids:
            continue
        class_to_indices.setdefault(target, []).append(idx)
    if not class_to_indices:
        return []

    per_class_limit = max(1, subset_size // len(class_to_indices))
    selected = []
    for indices in class_to_indices.values():
        rng.shuffle(indices)
        selected.extend(indices[:per_class_limit])
    if len(selected) > subset_size:
        selected = selected[:subset_size]
    return selected


def build_transform(is_train, cfg):
    mean, std = _get_mean_std(cfg)
    input_size = int(cfg["input_size"])
    resize_im = input_size > 32
    fixed_shift = cfg.get("fixed_post_norm_shift", 1.860)

    ops = []
    if is_train:
        if resize_im:
            ops.append(
                transforms.Resize(
                    (input_size, input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            )
        else:
            ops.append(transforms.RandomCrop(input_size, padding=4))
    else:
        if resize_im:
            if input_size >= 384:
                ops.append(
                    transforms.Resize(
                        (input_size, input_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    )
                )
            else:
                crop_pct = cfg.get("crop_pct", 224 / 256)
                size = int(input_size / crop_pct)
                ops.append(
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BICUBIC
                    )
                )
                ops.append(transforms.CenterCrop(input_size))

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ShiftMinToZero(fixed_shift=fixed_shift),
        ]
    )
    return transforms.Compose(ops)


def build_imagenet_dataset(is_train, cfg):
    split = "train" if is_train else "val"
    root = Path(cfg["data_root"]) / split
    base_dataset = datasets.ImageFolder(root, transform=build_transform(is_train, cfg))

    subset_size_key = "train_subset_size" if is_train else "val_subset_size"
    subset_size = cfg.get(subset_size_key)
    per_class = cfg.get("subset_per_class", True)
    seed = cfg.get("subset_seed", 42)
    class_ids = _resolve_class_ids(base_dataset, cfg, seed)
    indices = _select_subset_indices(
        base_dataset=base_dataset,
        subset_size=subset_size,
        seed=seed,
        per_class=per_class,
        class_ids=class_ids,
    )
    return ImageFolderAsDict(Subset(base_dataset, indices))


def build_imagenet_data(cfg):
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)

    split_from_train = bool(cfg.get("split_from_train", False))
    val_split = float(cfg.get("val_split", 0.0) or 0.0)
    split_seed = int(cfg.get("split_seed", 42))
    val_dataset = None

    if split_from_train and val_split > 0.0:
        train_root = Path(cfg["data_root"]) / "train"
        base_train = datasets.ImageFolder(
            train_root, transform=build_transform(True, cfg)
        )
        seed = cfg.get("subset_seed", 42)
        indices = _select_subset_indices(
            base_dataset=base_train,
            subset_size=cfg.get("train_subset_size"),
            seed=seed,
            per_class=cfg.get("subset_per_class", True),
            class_ids=_resolve_class_ids(base_train, cfg, seed),
        )
        rng = random.Random(split_seed)
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_split))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]

        train_dataset = ImageFolderAsDict(Subset(base_train, train_idx))
        val_base = datasets.ImageFolder(
            train_root, transform=build_transform(False, cfg)
        )
        val_dataset = ImageFolderAsDict(Subset(val_base, val_idx))
    else:
        train_dataset = build_imagenet_dataset(is_train=True, cfg=cfg)

    train_data = {
        "dataset": train_dataset,
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": cfg["train"]["num_workers"],
    }

    val_loaders = {}
    if cfg.get("val") and cfg["val"].get("batch_size"):
        if val_dataset is None:
            val_dataset = build_imagenet_dataset(is_train=False, cfg=cfg)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["val"]["batch_size"],
            shuffle=False,
            num_workers=cfg["val"]["num_workers"],
            drop_last=False,
        )
        evaluators = [
            MAEevaluatorIN(
                "results/imagenet",
                imagenet_default_mean_and_std=cfg.get(
                    "imagenet_default_mean_and_std", True
                ),
                fixed_post_norm_shift=cfg.get("fixed_post_norm_shift", 1.860),
            )
        ]
        val_loaders = {
            "IN_val": {
                "interval": cfg["val"].get("global_eval_interval", 1),
                "loader": val_loader,
                "evaluators": evaluators,
            }
        }

    return train_data, val_loaders
