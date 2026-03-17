import torch.nn as nn


from src.nets.utils import GRN
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from timm.models.layers import DropPath


class ConvNeXtV2Block2D(nn.Module):
    """
    ConvNeXtV2 block implementation.
    Args:
        dim (int): Number of input channels
        drop_path (float): Stochastic depth rate.
    Based on: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
    Note: we don't perform pemutations here because we implemented GRN differently.
    """

    def __init__(self, dim, drop_path=0.0):
        super(ConvNeXtV2Block2D, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, bias=True, groups=dim)
        self.norm = nn.GroupNorm(1, dim, 1e-6)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1, 1, 0)  # Equivalent to a Linear Layer
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1, 1, 0)  # Equivalent to a Linear Layer
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = identity + self.drop_path(x)
        return x
