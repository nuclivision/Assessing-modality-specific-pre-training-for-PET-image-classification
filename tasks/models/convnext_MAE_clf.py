import torch
import torch.nn as nn
import timm
import sys
from pathlib import Path
import os

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import src.sparse.sparse_transform as sparse_ops

from src.nets import convnext
from src.nets.convnext import build_network_from_cfg

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
import matplotlib.pyplot as plt
import numpy as np
from timm.layers import NormMlpClassifierHead


class ConvNeXtMAEWithAttention(nn.Module):
    def __init__(
        self,
        net=None,
        num_classes=2,
        train_param_budget=0,
        block_training_budget=0,
        feature_dim=768,
        attn_hidden_dim=128,
        dropout_p=0.5,
        pretrained=True,
        backbone="checkpoint_path.pth",
        linearprobe=True,
        pretrain_source="mae",
        timm_backbone=None,
        apply_intensity_log=True,
    ):
        super().__init__()

        # ----------------------------
        # Load pretrained encoder
        # ----------------------------
        self.pretrain_source = (pretrain_source or "mae").lower()
        if self.pretrain_source not in {"mae", "timm"}:
            raise ValueError(
                f"Unknown pretrain_source='{self.pretrain_source}'. Use 'mae' or 'timm'."
            )

        self.net = net
        self.encoder = None
        self.feature_dim = feature_dim
        self.linearprobe = linearprobe
        self.apply_intensity_log = bool(apply_intensity_log)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.pretrain_source == "mae":
            if self.net is None:
                raise ValueError("`net` must be provided when pretrain_source='mae'.")
            self.encoder = getattr(self.net, "network", self.net)
            self.feature_dim = self.encoder.dims[-1]
            if hasattr(self.encoder, "patch_size"):
                self.downsample_ratio = self.encoder.patch_size
            if pretrained:
                if not backbone:
                    raise ValueError(
                        "`model_args.backbone` must point to your MAE checkpoint when pretrain_source='mae'."
                    )
                ckpt = torch.load(backbone, map_location=device)
                state = ckpt.get("state_dict", ckpt)
                state = {
                    k: v
                    for k, v in state.items()
                    if not k.startswith(
                        ("reconstruction.", "decoder.", "pred.", "proj.", "mask_token")
                    )
                }
                self.encoder.load_state_dict(state, strict=False)
        else:
            backbone_name = timm_backbone or backbone
            if not backbone_name:
                raise ValueError(
                    "`model_args.timm_backbone` (or `backbone`) must be set when pretrain_source='timm'."
                )
            self.encoder = timm.create_model(backbone_name, pretrained=pretrained)
            inferred_dim = getattr(self.encoder, "num_features", None)
            if inferred_dim is None and hasattr(self.encoder, "head"):
                inferred_dim = getattr(self.encoder.head, "in_features", None)
            if inferred_dim is None:
                raise ValueError(
                    f"Could not infer feature dimension for timm backbone '{backbone_name}'."
                )
            self.feature_dim = inferred_dim

        self.train_param_budget = train_param_budget
        self.block_training_budget = block_training_budget

        # ----------------------------
        # Attention module
        # ----------------------------
        self.attn_fc1 = nn.Linear(self.feature_dim, attn_hidden_dim)
        self.attn_drop = nn.Dropout(dropout_p)
        self.attn_fc2 = nn.Linear(attn_hidden_dim, 1)

        # ----------------------------
        # Classifier
        # ----------------------------
        self.pre_cls_drop = nn.Dropout(dropout_p)
        self.classifier = NormMlpClassifierHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_size=192,
            pool_type="max",
            drop_rate=0.2,
        )
        if linearprobe:
            print("Using the linear probe")
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        params = 0
        for p in self.parameters():
            p.requires_grad = False
        for module in (self.attn_fc1, self.attn_fc2, self.classifier):
            for p in module.parameters():
                p.requires_grad = True
                params += p.numel()
        unfrozen_blocks = []
        all_blocks = self._iter_encoder_blocks()
        for block in reversed(all_blocks):
            if len(unfrozen_blocks) >= self.block_training_budget:
                break
            for p in block.parameters():
                p.requires_grad = True
            unfrozen_blocks.append(block)
        print("We just unfroze", len(unfrozen_blocks), "blocks out of", len(all_blocks), ". ")
        self.encoder_in_chans = self._infer_encoder_in_chans(default=1)

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
    
    def _iter_encoder_blocks(self):
        if not hasattr(self.encoder, "stages"):
            return []
        blocks = []
        for stage in self.encoder.stages:
            if hasattr(stage, "blocks"):
                blocks.extend(list(stage.blocks))
            else:
                blocks.extend(list(stage))
        return blocks

    def _infer_encoder_in_chans(self, default=1):
        in_chans = getattr(self.encoder, "in_chans", None)
        if isinstance(in_chans, int):
            return in_chans
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                return module.in_channels
        return default

    def _pool_features(self, feats):
        if self.pretrain_source == "timm" and hasattr(self.encoder, "forward_head"):
            pooled = self.encoder.forward_head(feats, pre_logits=True)
            if pooled.ndim == 4:
                pooled = pooled.mean(dim=(-2, -1))
            return pooled
        return feats.mean(dim=(-2, -1))

    def forward(self, x):
        B, C, num_mips, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * num_mips, C, H, W)
        if self.apply_intensity_log:
            x = torch.log1p(torch.clamp_min(x, 0))
        if C == 1 and self.encoder_in_chans == 3:
            x = x.repeat(1, 3, 1, 1)
        with torch.set_grad_enabled(self.training):
            if self.pretrain_source == "mae":
                x = self.encoder.downsample_layers[0](x)
            feats = self.encoder.forward_features(x)

        B_f, C_f, H_f, W_f = feats.shape
        feats_map = feats.view(B, num_mips, C_f, H_f, W_f)
        feats_pooled = self._pool_features(feats).view(B, num_mips, -1)
        attn_scores = torch.tanh(self.attn_fc1(feats_pooled))
        attn_scores = self.attn_drop(attn_scores)
        attn_scores = self.attn_fc2(attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=1)
        if not self.linearprobe:
            attn_weights = attn_weights.view(B, num_mips, 1, 1, 1)
            weighted_feats = torch.sum(attn_weights * feats_map, dim=1)
        else:
            weighted_feats = torch.sum(attn_weights * feats_pooled, dim=1)
        return self.classifier(self.pre_cls_drop(weighted_feats))


def build_classifier_mae(cfg):
    model_args = cfg.get("model_args", {})
    pretrain_source = model_args.get("pretrain_source", "mae")
    pretrained = model_args.get("pretrained", True)
    backbone = model_args.get("backbone")
    timm_backbone = model_args.get("timm_backbone")
    train_param_budget = model_args.get("train_param_budget")
    block_training_budget = model_args.get("block_training_budget")
    linearprobe = model_args.get("linearprobe")
    apply_intensity_log = model_args.get("apply_intensity_log", True)

    net = None
    if pretrain_source.lower() == "mae":
        net = build_network_from_cfg(cfg["args"]["network"])
    else:
        print(
            "Using timm pretrained ConvNeXtV2 backbone:",
            timm_backbone or backbone,
        )

    my_clf = ConvNeXtMAEWithAttention(
        net=net,
        train_param_budget=train_param_budget,
        pretrained=pretrained,
        backbone=backbone,
        pretrain_source=pretrain_source,
        timm_backbone=timm_backbone,
        linearprobe=linearprobe,
        block_training_budget=block_training_budget,
        apply_intensity_log=apply_intensity_log,
    )

    total_params = my_clf.count_parameters(exclude=("reconstruction",))
    encoder_param_ids = {id(p) for p in my_clf.encoder.parameters()}
    non_encoder_params = sum(
        p.numel() for _, p in my_clf.named_parameters() if id(p) not in encoder_param_ids
    )
    print(f"Total model parameters: {total_params:,}")
    print(f"Non-encoder parameters: {non_encoder_params:,}")

    return my_clf


build_MAE_clf = build_classifier_mae
