import torch
import torch.nn as nn
import torchvision.models as models


class ResNet_MIP_classifier(nn.Module):

    def __init__(
        self,
        num_classes,
        n_input_channels=1,
        reduced_dim=256,
        attn_hidden_dim=128,
        dropout_p=0.5,
        pretrained=True,
        resnet="resnet50",
        apply_intensity_log=False,
    ):
        super().__init__()
        self.apply_intensity_log = bool(apply_intensity_log)
        if resnet == "resnet18":
            base = models.resnet18(pretrained=pretrained)
        if resnet == "resnet50":
            base = models.resnet50(pretrained=pretrained)

            # rewrite conv1 if needed
        if n_input_channels != 3:
            old_conv1 = base.conv1
            base.conv1 = nn.Conv2d(
                n_input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            with torch.no_grad():
                if n_input_channels == 1:
                    base.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
                else:
                    avg = old_conv1.weight.mean(dim=1, keepdim=True)
                    base.conv1.weight.copy_(avg.repeat(1, n_input_channels, 1, 1))

        self.encoder = nn.Sequential(*list(base.children())[:-1])

        # attention + classifier identical to original head
        self.attn_fc1 = nn.Linear(2048, attn_hidden_dim)
        self.attn_drop = nn.Dropout(dropout_p)
        self.attn_fc2 = nn.Linear(attn_hidden_dim, 1)

        self.pre_cls_drop = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(2048, num_classes)

        print("Number of trainable parameters:", self._count_trainable_params())

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, D, C, H, W)
        x = x.view(B * D, C, H, W)
        if self.apply_intensity_log:
            x = torch.log1p(torch.clamp_min(x, 0))

        feats = self.encoder(x)
        feats = torch.flatten(feats, 1)

        feats = feats.view(B, D, -1)  # pack slices for attention: (B, D, reduced_dim)

        attn_scores = torch.tanh(self.attn_fc1(feats))
        attn_scores = self.attn_drop(attn_scores)
        attn_scores = self.attn_fc2(attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * feats, dim=1)

        logits = self.classifier(self.pre_cls_drop(pooled))
        return logits

    def _count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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


def build_resnet_scratch(cfg):
    model_args = cfg.get("model_args", {})
    num_classes = model_args.get("num_classes", 2)
    pretrained = model_args.get("pretrained", True)
    n_input_channels = model_args.get("n_input_channels", 1)
    reduced_dim = model_args.get("reduced_dim", 256)
    attn_hidden_dim = model_args.get("attn_hidden_dim", 128)
    dropout_p = model_args.get("dropout_p", 0.5)
    resnet = model_args.get("resnet", "resnet50")
    apply_intensity_log = model_args.get("apply_intensity_log", False)

    my_clf = ResNet_MIP_classifier(
        num_classes=num_classes,
        n_input_channels=n_input_channels,
        reduced_dim=reduced_dim,
        attn_hidden_dim=attn_hidden_dim,
        dropout_p=dropout_p,
        pretrained=pretrained,
        resnet=resnet,
        apply_intensity_log=apply_intensity_log,
    )

    total_params = my_clf.count_parameters(exclude=("reconstruction",))
    print(f"Total model parameters: {total_params:,}")

    return my_clf


build_MAE_clf = build_resnet_scratch
