# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)



from collections import OrderedDict
from models.augment_stage import AuxiliaryHead
import torch.nn as nn
from models import ops
from models.augment_cell import AugmentCell
from models.augment_stage_imagenet import AuxiliaryHeadImagenet
from utils import setting


class AugmentCellCNN(nn.Module):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype, stem_multiplier=3):
        super().__init__()
        self.C_in = C_in
        self.C = C                   # 36
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        C_cur = stem_multiplier * C
        if input_size == setting.IMAGENET_SIZE:
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C_cur, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_cur),
            )
            self.stem = nn.Sequential(
                OrderedDict(
                    [
                        ("stem0", self.stem0),
                        ("stem1", self.stem1)
                    ]
                )
            )
        else:
            self.stem0 = nn.Sequential(
                nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
                nn.BatchNorm2d(C_cur)
            )
            self.stem = nn.Sequential(
                OrderedDict(
                    [
                        ("stem0", self.stem0),
                    ]
                )
            )

        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
        
            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction, i, n_layers)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                if input_size == setting.IMAGENET_SIZE:
                    self.aux_head = AuxiliaryHeadImagenet(14, 16 * C, n_classes)
                else:
                    self.aux_head = AuxiliaryHead(input_size // 4, C_p, n_classes)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
    
    def forward(self, x):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)
        
        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits, aux_logits
    
    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
