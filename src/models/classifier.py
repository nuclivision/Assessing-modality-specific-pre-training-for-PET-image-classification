import torch
import torch.nn as nn
import timm
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import src.nets.sparse_transform as sparse_ops
from src.nets.convnext import build_network_from_cfg
from timm.layers import NormMlpClassifierHead


class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        net=None,
        num_classes=2,
        block_training_budget=0,
        feature_dim=768,
        attn_hidden_dim=128,
        dropout_p=0.5,
        pretrained=True,
        backbone=None,
        linearprobe=True,
        pretrain_source="mae",
        apply_intensity_log=True,
    ):
        super().__init__()

        self.pretrain_source = (pretrain_source or "mae").lower()
        if self.pretrain_source not in {"mae", "timm"}:
            raise ValueError(
                f"Unknown pretrain_source='{self.pretrain_source}'. Use 'mae' or 'timm'."
            )

        self.linearprobe = linearprobe
        self.apply_intensity_log = bool(apply_intensity_log)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.pretrain_source == "mae":
            if net is None:
                raise ValueError(
                    "`net` must be provided for a MAE checkpoint backbone."
                )
            self.encoder = getattr(net, "network", net)
            self.feature_dim = self.encoder.dims[-1]
            if pretrained:
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
            if not backbone:
                raise ValueError(
                    "`backbone` must be a timm model name when not a checkpoint path."
                )
            self.encoder = timm.create_model(backbone, pretrained=pretrained)
            inferred_dim = getattr(self.encoder, "num_features", None)
            if inferred_dim is None and hasattr(self.encoder, "head"):
                inferred_dim = getattr(self.encoder.head, "in_features", None)
            if inferred_dim is None:
                raise ValueError(
                    f"Could not infer feature dimension for timm backbone '{backbone}'."
                )
            self.feature_dim = inferred_dim

        self.block_training_budget = block_training_budget

        self.attn_fc1 = nn.Linear(self.feature_dim, attn_hidden_dim)
        self.attn_drop = nn.Dropout(dropout_p)
        self.attn_fc2 = nn.Linear(attn_hidden_dim, 1)

        self.pre_cls_drop = nn.Dropout(dropout_p)
        if linearprobe:
            print("Using linear probe")
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = NormMlpClassifierHead(
                in_features=self.feature_dim,
                num_classes=num_classes,
                hidden_size=192,
                pool_type="max",
                drop_rate=0.2,
            )

        for p in self.parameters():
            p.requires_grad = False
        for module in (self.attn_fc1, self.attn_fc2, self.classifier):
            for p in module.parameters():
                p.requires_grad = True

        all_blocks = self._iter_encoder_blocks()
        unfrozen_blocks = []
        for block in reversed(all_blocks):
            if len(unfrozen_blocks) >= self.block_training_budget:
                break
            for p in block.parameters():
                p.requires_grad = True
            unfrozen_blocks.append(block)
        print(f"Unfroze {len(unfrozen_blocks)} / {len(all_blocks)} encoder blocks.")

        self.encoder_in_chans = self._infer_encoder_in_chans(default=1)

    def _iter_encoder_blocks(self):
        if not hasattr(self.encoder, "stages"):
            return []
        blocks = []
        for stage in self.encoder.stages:
            blocks.extend(
                list(stage.blocks) if hasattr(stage, "blocks") else list(stage)
            )
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
        x = x.permute(0, 2, 1, 3, 4).reshape(B * num_mips, C, H, W)

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

        attn_scores = self.attn_fc2(
            self.attn_drop(torch.tanh(self.attn_fc1(feats_pooled)))
        )
        attn_weights = torch.softmax(attn_scores, dim=1)

        if self.linearprobe:
            weighted_feats = torch.sum(attn_weights * feats_pooled, dim=1)
        else:
            weighted_feats = torch.sum(
                attn_weights.view(B, num_mips, 1, 1, 1) * feats_map, dim=1
            )

        return self.classifier(self.pre_cls_drop(weighted_feats))


def build_classifier(cfg):
    model_args = cfg.get("model_args", {})
    pretrained = model_args.get("pretrained", True)
    backbone = model_args.get("backbone")
    block_training_budget = model_args.get("block_training_budget", 0)
    linearprobe = model_args.get("linearprobe", True)
    apply_intensity_log = model_args.get("apply_intensity_log", True)

    pretrain_source = model_args.get("pretrain_source", "mae")
    net = (
        build_network_from_cfg(cfg["args"]["network"])
        if pretrain_source.lower() == "mae"
        else None
    )
    if pretrain_source.lower() == "timm":
        print(f"Using timm pretrained backbone: {backbone}")

    model = ConvNeXtClassifier(
        net=net,
        pretrained=pretrained,
        backbone=backbone,
        pretrain_source=pretrain_source,
        linearprobe=linearprobe,
        block_training_budget=block_training_budget,
        apply_intensity_log=apply_intensity_log,
    )

    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    total_params = sum(p.numel() for p in model.parameters())
    non_encoder_params = sum(
        p.numel() for p in model.parameters() if id(p) not in encoder_param_ids
    )
    print(f"Total model parameters: {total_params:,}")
    print(f"Non-encoder parameters: {non_encoder_params:,}")

    return model


build_MAE_clf = build_classifier
