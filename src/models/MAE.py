from __future__ import annotations


print("Loading MAE preprocessor...")
print("Importing necessary modules...")

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nucli_train.models.builders import build_model
from nucli_train.models.image_translation import ImageTranslationModel
import src.sparse.sparse_transform as sparse_ops

import torch
import torch.nn as nn

import src.nets.convnext
from src.nets.convnext import build_network_from_cfg

torch.set_printoptions(threshold=torch.inf)  # no truncation
torch.set_printoptions(linewidth=200)


class MIM(ImageTranslationModel):

    def __init__(
        self,
        net,
        mask_ratio=0.6,
        loss_functions=None,
        patch_size=None,
        intensitylog=False,
        prepretrain=False,
        **kwargs,
    ):
        super().__init__(net=net, loss_functions=loss_functions, **kwargs)

        self.network = net
        self.base_network = getattr(net, "network", net)
        self.intensitylog = intensitylog
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if patch_size is None:
            if hasattr(self.base_network, "patch_size"):
                patch_size = self.base_network.patch_size
            else:
                raise ValueError(
                    "patch_size must be provided when the network lacks a patch_size attribute."
                )
        self.prepretrain = prepretrain
        self.patch_size = patch_size
        if hasattr(self.base_network, "mask_ratio"):
            print("using the mask ratio from the net")
            mask_ratio = self.base_network.mask_ratio
        print(f"Using a mask ratio of {mask_ratio}")
        self.mask_ratio = mask_ratio
        self.downsample_ratio = 1
        for layer in self.base_network.downsample_layers:
            if isinstance(layer, nn.Sequential):
                conv = next((m for m in layer if isinstance(m, nn.Conv2d)), None)
            else:
                conv = (
                    layer[0] if isinstance(layer, (nn.Conv2d, nn.Sequential)) else None
                )
            if conv is not None:
                self.downsample_ratio *= conv.stride[0]
        print("Downsample ratio:", self.downsample_ratio)
        if hasattr(self.network, "get_optimizer"):
            self.opt = self.network.get_optimizer()
        elif hasattr(self.base_network, "get_optimizer"):
            self.opt = self.base_network.get_optimizer()
        else:
            self.opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.base_network.parameters())
            )
        if hasattr(self.base_network, "prepretrained_ckpt"):
            self.prepretrained_ckpt = self.base_network.prepretrained_ckpt
            if self.prepretrained_ckpt is not None and self.prepretrained_ckpt != "":
                ckpt = torch.load(self.prepretrained_ckpt, map_location=device)
                state = ckpt.get("state_dict", ckpt)
                missing, unexpected = self.base_network.load_state_dict(
                    state, strict=False
                )
                print("Loaded phase 1 ckpt:", self.prepretrained_ckpt)
                print(
                    "Missing keys:", len(missing), "Unexpected keys:", len(unexpected)
                )

    def count_parameters(self, include=None, exclude=None, trainable_only=False):
        include = tuple(include or [])
        exclude = tuple(exclude or [])
        total = 0
        for name, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if include and not name.startswith(include):
                continue
            if exclude and name.startswith(exclude):
                continue
            total += p.numel()
        return total

    def _get_expected_in_chans(self):
        # Prefer network attribute when available.
        in_chans = getattr(self.base_network, "in_chans", None)
        if isinstance(in_chans, int) and in_chans > 0:
            return in_chans

        # Fallback: infer from the first Conv2d.
        for module in self.base_network.modules():
            if isinstance(module, nn.Conv2d):
                return int(module.in_channels)
        return None

    def _adapt_input_channels(self, x):
        expected = self._get_expected_in_chans()
        if expected is None:
            return x
        if x.ndim != 4:
            return x

        c_in = int(x.shape[1])
        if c_in == expected:
            return x
        if c_in == 1 and expected > 1:
            return x.repeat(1, expected, 1, 1)
        if c_in > expected:
            return x[:, :expected, ...]

        raise ValueError(
            f"Cannot adapt input channels: got {c_in}, expected {expected}."
        )

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H % p == 0 and W % p == 0
        h, w = H // p, W // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5) 
        x = x.reshape(B, h * w, C * p * p) 
        return x

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 * 1)
        imgs: (B, C, H, W)
        """
        p = self.patch_size
        B, L, patch_dim = x.shape
        h = w = int(L**0.5 + 1e-5)
        assert h * w == L
        C = patch_dim // (p * p)

        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  
        imgs = x.reshape(B, C, h * p, w * p)
        return imgs

    def upsample_mask(self, mask, h, w, scale_h, scale_w):
        mask = mask.reshape(-1, h, w)  
        mask = mask.repeat_interleave(scale_h, dim=1)
        mask = mask.repeat_interleave(scale_w, dim=2)
        return mask

    def downsample_mask(self, mask, scale):
        B, C, H, W = mask.shape

        assert H % scale == 0

        h = H // scale

        mask = mask.reshape(B, C, h, scale, h, scale)
        mask = mask.any(dim=3).any(dim=4)
        mask = mask.float()
        return mask 

    def gen_random_mask(self, x, mask_ratio):
        B, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size
        L = h * w  
        patches = self.patchify(x)
        eligible = patches.abs().sum(-1) != 0
        num_eligible = eligible.sum(dim=1)
        len_keep_eligible = (num_eligible * (1 - mask_ratio)).long()

        noise = torch.rand(B, L, device=x.device)
        noise[~eligible] = 2.0
        ids_shuffle = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids_shuffle, dim=1)

        keep_eligible = ranks < len_keep_eligible.unsqueeze(1)
        mask = keep_eligible.float()
        return mask, h, w 

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*2]
        mask: [N, L], 1 is remove, 0 is keep
        """
        losses = {}

        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum("ncl->nlc", pred)

        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  

        eligible = (target.abs().sum(dim=-1) != 0).float()
        removed_eligible = (1 - mask) * eligible
        denom = removed_eligible.sum().clamp_min(1.0)
        loss = (loss * removed_eligible).sum() / denom
        losses["value"] = loss
        return losses

    def train_step(self, batch):
        input_data = batch["data"].cuda()
        if self.intensitylog:
            print("Using log(1+x) in training!")
            input_data = torch.log1p(torch.clamp_min(input_data, 0))
        if self.prepretrain == "IN":
            input_data = input_data.repeat(1, 3, 1, 1)
        input_data = self._adapt_input_channels(input_data)
        B, C, H, W = input_data.shape
        patch_mask_long, h, w = self.gen_random_mask(input_data, self.mask_ratio)
        patch_mask = patch_mask_long.view(B, 1, h, w)
        f_h = H // self.downsample_ratio
        active_b1ff = torch.nn.functional.interpolate(
            patch_mask.float(), size=(f_h, f_h), mode="nearest"
        ).bool()
        sparse_ops._cur_active = active_b1ff
        x = self.base_network.downsample_layers[0](
            input_data
        )  
        H, W = x.shape[-2], x.shape[-1]
        scale = H // h 
        mask = self.upsample_mask(patch_mask, h, w, scale, scale)
        mask = mask.unsqueeze(1).type_as(x) 
        x *= mask  
        b, _, h_mask, w_mask = mask.shape
        h_dec = input_data.shape[-1] // self.downsample_ratio
        scale_h = h_mask // h_dec
        mask_dec = self.downsample_mask(mask, scale_h)
        preds, feats = self.base_network.forward(x, mask_dec) 
        targets = input_data 
        losses = self.forward_loss(targets, preds, patch_mask_long)
        return losses

    def validation_step(self, batch):
        with torch.no_grad():
            input_data = batch["data"].cuda()
            if self.intensitylog:
                print("Using log(1+x) in validation!")
                input_data = torch.log1p(torch.clamp_min(input_data, 0))
            if self.prepretrain == "IN":
                input_data = input_data.repeat(1, 3, 1, 1)
            input_data = self._adapt_input_channels(input_data)
            pid = batch["patient_id"]
            B, C, H, W = input_data.shape
            patch_mask, h, w = self.gen_random_mask(input_data, self.mask_ratio)
            patch_mask_long = patch_mask.clone()
            patch_mask_expanded = patch_mask.unsqueeze(-1).repeat(
                1, 1, self.patch_size * self.patch_size
            )  

            voxel_mask = self.unpatchify(patch_mask_expanded)  # (B, 1, H, W)
            patch_mask = patch_mask_long.view(B, 1, h, w)
            f_h = H // self.downsample_ratio
            active_b1ff = torch.nn.functional.interpolate(
                patch_mask.float(), size=(f_h, f_h), mode="nearest"
            ).bool()
            sparse_ops._cur_active = active_b1ff
            x = self.base_network.downsample_layers[0](input_data)
            H, W = x.shape[-2], x.shape[-1]
            scale = H // h 
            mask = self.upsample_mask(patch_mask, h, w, scale, scale)
            mask = mask.unsqueeze(1).type_as(x)
            x *= mask 
            b, _, h_mask, w_mask = mask.shape  
            h_dec = (
                input_data.shape[-1] // self.downsample_ratio
            ) 
            scale_h = h_mask // h_dec
            mask_dec = self.downsample_mask(mask, scale_h)
            preds, feats = self.base_network.forward(x, mask_dec)  
            losses = self.forward_loss(input_data, preds, patch_mask_long)
            ##################### EVALUATION  ######################
            B, _, H, W = preds.shape  # e.g., 6, 1024, 6, 6

            if W == input_data.shape[-1]:
                preds_eval = preds
            else:
                preds_flat = preds.permute(0, 2, 3, 1).reshape(B, H * W, -1)
                preds_eval = self.unpatchify(preds_flat)
        return {
            "losses": losses,
            "predictions": preds_eval,
            "input": input_data,
            "mask": voxel_mask,
            "batch_size": input_data.shape[0],
            "masked_data": input_data * voxel_mask,
            "intensitylog": self.intensitylog,
            "patient_id": pid,
            "feats": feats,
        }

    def count_params_by_bucket(self, buckets):
        counts = {b: 0 for b in buckets}
        counts["other"] = 0
        for name, p in self.named_parameters():
            matched = False
            for b in buckets:
                if b in name:
                    counts[b] += p.numel()
                    matched = True
                    break
            if not matched:
                counts["other"] += p.numel()
        return counts


def build_mae_model(cfg):
    net = build_network_from_cfg(cfg["args"]["network"])
    losses = cfg["args"].get("losses", None)

    patch_size = cfg["args"]["patch_size"]
    intensitylog = cfg["args"].get("intensitylog", False)
    prepretrain = cfg["args"].get("prepretrain", False)

    my_MIM = MIM(
        net=net,
        loss_functions=losses,
        patch_size=patch_size,
        intensitylog=intensitylog,
        prepretrain=prepretrain,
    )

    total_params = my_MIM.count_parameters(exclude=("network.network.reconstruction",))
    buckets = [
        "reconstruction",
        "decoder",
        "pred",
        "proj",
        "downsample_layers",
        "stages",
    ]
    params_per_bucket = my_MIM.count_params_by_bucket(buckets)
    encoder_only = my_MIM.count_parameters(
        include=(
            "network.network.stages",
            "network.network.downsample_layers",
        )
    )
    
    print("Params encoder only:", encoder_only)
    print(f"Total model parameters: {total_params:,}")
    print("Params by bucket:", params_per_bucket)

    return my_MIM


build_MAE = build_mae_model
