# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from pathlib import Path
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]  # repo root above scripts/
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

_cur_active: torch.Tensor = None  # B1ff

def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(
        w_repeat, dim=3
    )
    return (
        active_ex
        if returning_active_ex
        else active_ex.squeeze(1).nonzero(as_tuple=True)
    )  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(
        H=x.shape[2], W=x.shape[3], returning_active_ex=True
    )  # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_conv_forward_reweighted_optimized(self, x: torch.Tensor):
    """
    Optimized version using grouped convolution trick for normalization.
    """
    B, C_in, H_in, W_in = x.shape

    input_mask = _get_active_ex_or_ii(H=H_in, W=W_in, returning_active_ex=True).type_as(
        x
    )  

    # Step 1: Apply convolution to masked input
    x_masked = x * input_mask
    output = super(type(self), self).forward(x_masked)

    if self.bias is not None:
        output_no_bias = output - self.bias.view(1, -1, 1, 1)
    else:
        output_no_bias = output

    # Step 2: Compute normalization (optimized)
    input_mask_expanded = input_mask.expand(-1, C_in, -1, -1) 

    kernel_magnitude = self.weight.abs()  

    mask_sum = F.conv2d(
        input_mask_expanded,
        kernel_magnitude,
        bias=None,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups,
    )  
    # Step 3: Renormalize
    eps = 1e-5
    full_sum = self.weight.abs().sum(
        dim=(1, 2, 3)
    )  
    full_sum = full_sum.view(1, -1, 1, 1)
    output_normalized = output_no_bias / (mask_sum + eps) * full_sum
    if self.bias is not None:
        output_normalized += self.bias.view(1, -1, 1, 1)

    # Step 4: Update output mask
    output_mask = _get_active_ex_or_ii(
        H=output.shape[2], W=output.shape[3], returning_active_ex=True
    )
    output_normalized = output_normalized * output_mask

    return output_normalized


class SparseConv2dReweighted(nn.Conv2d):
    """Sparse Conv2d with proper reweighting"""

    forward = sp_conv_forward_reweighted_optimized


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)

    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[
        ii
    ]  # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(
        nc
    )  # use BN1d to normalize this flatten feature `nc`

    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True
    ):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 4:  # BHWC or BCHW
            if self.data_format == "channels_last":  # BHWC
                if self.sparse:
                    ii = _get_active_ex_or_ii(
                        H=x.shape[1], W=x.shape[2], returning_active_ex=False
                    )
                    nc = x[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else:  # channels_first, BCHW
                if self.sparse:
                    ii = _get_active_ex_or_ii(
                        H=x.shape[2], W=x.shape[3], returning_active_ex=False
                    )
                    bhwc = x.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    x = torch.zeros_like(bhwc)
                    x[ii] = nc
                    return x.permute(0, 3, 1, 2)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                    return x
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return (
            super(SparseConvNeXtLayerNorm, self).__repr__()[:-1]
            + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'
        )


class SparseConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        sparse=True,
        reweighted=True,
        ks=7,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=ks // 2, groups=dim)
        if sparse:
            self.dwconv = SparseConv2d(
                dim, dim, kernel_size=ks, padding=ks // 2, groups=dim
            )
            if reweighted:
                self.dwconv = SparseConv2dReweighted(
                    dim, dim, kernel_size=ks, padding=ks // 2, groups=dim
                )  
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path: nn.Module = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.sparse = sparse

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(
            x
        )  
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        if self.sparse:
            x *= _get_active_ex_or_ii(
                H=x.shape[2], W=x.shape[3], returning_active_ex=True
            )

        x = input + self.drop_path(x)
        return x

    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f", sp={self.sparse})"
