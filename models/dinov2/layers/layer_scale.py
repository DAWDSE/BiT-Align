# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, tuple) and len(x) == 3:
            x1, attn, key = x # 增加了attn和key的返回，因此x变成了长度为3的元组，这个函数要单独对x处理
            if self.inplace:
                x1 = x1.mul_(self.gamma)
            else:
                x1 = x1 * self.gamma
            return x1, attn, key
        else:
            return x.mul_(self.gamma) if self.inplace else x * self.gamma
        #return x1, attn, key
