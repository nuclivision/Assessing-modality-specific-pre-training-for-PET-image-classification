import torch
import argparse
import monai
import pandas as pd
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path.cwd().parent))  # point to repo root

from src.models.factory import build_local_model

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Lambdad,
    EnsureTyped,
    ToTensord,
)
from monai.data import Dataset
import src.sparse.sparse_transform as sparse_ops


def run_inference(setup_cfg_path, checkpoint_path, test_csv):
    setup = yaml.safe_load(open(setup_cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataframe
    df = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv.copy()
    print(len(df))
    print(df)
    records = [
        {
            "PatientID": row["PatientID"],
            "Label": row["Label"],
            "image": row["image"],
        }
        for row in df.to_dict("records")
    ]

    # Build inference transforms: deterministic part + ToTensord (no augmentation)
    deterministic = Compose(
        [
            LoadImaged(keys=("image")),
            EnsureChannelFirstd(keys=("image")),
            Lambdad(keys="image", func=lambda x: x.clip(max=10.0)),
            EnsureTyped(keys=("image", "Label")),
        ]
    )
    val_transforms = Compose([deterministic, ToTensord(keys=("image", "Label"))])

    dataset = Dataset(data=records, transform=val_transforms)
    loader = DataLoader(
        dataset, batch_size=setup["dataloader"]["batch_size"], shuffle=False
    )

    # Load model + weights
    model_cfg = setup["model"]
    model = build_local_model(model_cfg).to(device)
    print("checkpoint path:", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    print("keys of ckpt:", ckpt.keys())
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("length of dataset:", len(dataset))
    preds, labels, patient_ids, prob_pos = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            outputs = model(inputs)
            softmax_outputs = torch.softmax(outputs, dim=1)
            conf, pred = softmax_outputs.max(dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(batch["Label"].numpy())
            prob_pos.extend(softmax_outputs[:, 1].cpu().numpy())
            patient_ids.extend(batch["PatientID"])

    results_df = pd.DataFrame(
        {
            "PatientID": patient_ids,
            "Label": labels,
            "Pred": preds,
            "Prob_Pos": prob_pos,
        }
    )
    return results_df


def run_mae_inference(setup_cfg_path, checkpoint_path, test_csv):
    setup = yaml.safe_load(open(setup_cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv.copy()

    records = [
        {
            "PatientID": row["PatientID"],
            "Label": row["Label"],
            "image": row["image"],
        }
        for row in df.to_dict("records")
    ]

    # Load model + weights
    model_cfg = setup["model"]
    model = build_local_model(model_cfg).to(device)
    print("checkpoint path:", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    # add prefix so keys match model
    new_state = {}
    for k, v in state.items():
        if k.startswith("network."):
            new_state["network." + k] = v
        else:
            new_state["network.network." + k] = v

    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in new_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    print("filtered:", len(filtered))
    print("checkpoint key sample:", list(state.keys())[:5])
    print("model key sample:", list(model_state.keys())[:5])

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("missing:", missing)
    print("unexpected:", len(unexpected))
    # model.load_state_dict(new_state, strict=True)
    # print("checkpoint keys:", ckpt.keys())
    # model.load_state_dict(ckpt)
    model.eval()

    test_ds = NiftiDataset(records, transform=None)
    print("test ds 1:", test_ds[0]["data"].shape)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    batch = next(iter(test_loader))
    out = mae_inference_step(model, batch, mask_ratio=0.7)

    print(out["input"].shape, out["preds"].shape)

    return out["input"], out["preds"], out["mask"]


def mae_inference_step(model, batch, mask_ratio=None):
    model.eval()
    input_data = batch["data"].cuda()  # or .to(device)
    B, C, nb_MIPs, H, W = input_data.shape
    input_data = input_data.reshape(B * nb_MIPs, C, H, W)
    B, C, H, W = input_data.shape
    print("shape of the image:", input_data.shape)

    # Use provided mask_ratio or model default
    ratio = mask_ratio if mask_ratio is not None else model.mask_ratio

    patch_mask_long, h, w = model.gen_random_mask(input_data, ratio)
    print("shape patch mask long:", patch_mask_long.shape)
    patch_mask = patch_mask_long.view(B, 1, h, w)

    f_h = H // model.downsample_ratio
    active_b1ff = torch.nn.functional.interpolate(
        patch_mask.float(), size=(f_h, f_h), mode="nearest"
    ).bool()
    sparse_ops._cur_active = active_b1ff
    print("mean x before stem", input_data.mean())

    x = model.network.network.downsample_layers[0](input_data)
    print("mean x after stem:", x.mean())
    Hx, Wx = x.shape[-2], x.shape[-1]
    scale = Hx // h

    mask = model.upsample_mask(patch_mask, h, w, scale, scale)
    mask = mask.unsqueeze(1).type_as(x)
    print("mean x:", x.mean(), "mean masked x", (x * mask).mean())
    x = x * mask

    h_mask, w_mask = mask.shape[-2], mask.shape[-1]
    h_dec = input_data.shape[-1] // model.downsample_ratio
    scale_h = h_mask // h_dec
    mask_dec = model.downsample_mask(mask, scale_h)

    # feats, proj_preds = model.network.forward(x, mask_dec)

    preds, feats = model.network.forward(
        x, mask_dec
    )  # preds already reconstructed if reconstruct=True

    # if preds are already image-space, skip unpatchify
    if preds.shape[-1] == input_data.shape[-1]:
        preds_eval = preds
        print("shape of preds:", preds.shape)
    else:
        B, C, Hp, Wp = preds.shape
        preds_flat = preds.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
        preds_eval = model.unpatchify(preds_flat)
        print(
            "preds shape:",
            preds.shape,
            "flat shape:",
            preds_flat.shape,
            "eval shape:",
            preds_eval.shape,
        )

    # img_pred = model.unpatchify(preds)
    # print("shape preds:", preds.shape, "img pred shape:", img_pred.shape)
    return {
        "input": input_data,
        "preds": preds_eval,
        "mask": mask,
        "patch_mask_long": patch_mask_long,
    }


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class NiftiDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        vol = nib.load(r["image"]).get_fdata().astype(np.float32)

        # If channels last: [H,W,4] -> [4,H,W]
        if vol.shape[-1] == 4:
            mips = np.moveaxis(vol, -1, 0)
        # If channels first: [4,H,W] -> [4,H,W]
        elif vol.shape[0] == 4:
            mips = vol
        else:
            raise ValueError(f"Unexpected shape {vol.shape}, expected 4 MIP channels.")

        x = torch.from_numpy(mips).unsqueeze(0)  # [1,4,H,W]
        x = torch.log1p(x)
        if self.transform:
            x = self.transform(x)

        return {"data": x, "PatientID": r["PatientID"], "Label": r["Label"]}
