# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.augment_stage_imagenet import AuxiliaryHeadImagenet
from models.get_cell import Get_StageSpecified_Cell
from models.get_cell import GetCell
from models.get_dag import GetStage
from models import ops
from models.search_bigDAG import SearchBigDAG
from models.search_stage import SearchStage, SearchStageController, broadcast_list
from utils import setting
import genotypes.genotypes as gt



class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)
    
    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class AugmentStage(nn.Module):
    """" Augmented DAG-CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 DAG, stem_multiplier=4, cell_multiplier=4, spec_cell=False):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
            genotype: the struct of normal cell & reduce cell
            DAG: the struct of big-DAG
        """
        super().__init__()
        self.C_in = C_in
        self.C = C                   # 36
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.DAG = DAG
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
        C_pp, C_p, C_cur = cell_multiplier * C, cell_multiplier * C, C

        self.cells = nn.ModuleList()

        lenDAG1, lenDAG2, lenDAG3 = len(self.DAG.DAG1), len(self.DAG.DAG2), len(self.DAG.DAG3)
        self.aux_head = nn.Identity()

        for i in range(n_layers):
            # if i in [0,1,2,3,4,5]:
            if i in range(lenDAG1):
                reduction = False
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                # cell = GetCell(genotype, 4 * C, 4 * C, C, reduction) # out 144=4*C
                # 144 144 36  out=144  DAG_out=144*2=288
                self.cells.append(cell)
            # if i in [6]:
            if i in [lenDAG1]:
                self.bigDAG1 = GetStage(DAG, self.cells, 0, lenDAG1 - 1, stem_multiplier*C_cur, stem_multiplier*C_cur, cell_multiplier * C_cur)

                reduction = True
                C_pp = C_p = 2*cell_multiplier*C_cur
                C_cur *= 2
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                # cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction) # out 72*4=288
                # 288, 288, 72 out=288=4*72
                self.cells.append(cell)
            # if i in [7,8,9,10,11,12]:
            if i in range(lenDAG1 + 1, lenDAG1 + 1 + lenDAG2):
                reduction = False
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                # cell = GetCell(genotype, 8 * C, 8 * C, 2 * C, reduction) # out 288
                # 288, 288, 72, out=72*4=288  DAG_out=288*2=576
                self.cells.append(cell)
            # if i in [13]:
            if i in [lenDAG1 + 1 + lenDAG2]:
                self.bigDAG2 = GetStage(DAG, self.cells, lenDAG1 + 1, lenDAG1 + lenDAG2, C_pp, C_p, cell_multiplier * C_cur)

                reduction = True
                C_pp = C_p = 2*cell_multiplier*C_cur
                C_cur *= 2
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                # cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction) # out 144*4=576
                self.cells.append(cell)
            # if i in [14,15,16,17,18,19]:
            if i in range(lenDAG1 + 2 + lenDAG2, lenDAG1 + 2 + lenDAG2 + lenDAG3):
                reduction = False
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                # cell = GetCell(genotype, 16 * C, 16 * C, 4 * C, reduction) # out 144*4=576
                self.cells.append(cell)  # DAG_out=576*2=1152

            if i == self.aux_pos:
                if input_size == setting.IMAGENET_SIZE:
                    self.aux_head = AuxiliaryHeadImagenet(14, C_p, n_classes)
                else:
                    self.aux_head = AuxiliaryHead(input_size // 4, C_p, n_classes)

                C_pp, C_p = cell_multiplier*C_cur, cell_multiplier*C_cur
        
        # self.bigDAG1 = GetStage(DAG, self.cells, 0, lenDAG1 - 1, 4 * C, 4 * C, 4 * C)
        # self.bigDAG2 = GetStage(DAG, self.cells, lenDAG1 + 1, lenDAG1 + lenDAG2, 8 * C, 8 * C, 8 * C)
        self.bigDAG3 = GetStage(DAG, self.cells, lenDAG1 + 2 + lenDAG2, lenDAG1 + 1 + lenDAG2 + lenDAG3, C_pp, C_p, cell_multiplier * C_cur)      

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32 * C, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        aux_logits = None
        s0 = s1 = self.bigDAG1(s0, s1)
        s0 = s1 = self.cells[1 * self.n_layers // 3](s0, s1) # reduction
        s0 = s1 = self.bigDAG2(s0, s1) 
        s0 = s1 = self.cells[2 * self.n_layers // 3](s0, s1) # reduction
        aux_logits = self.aux_head(s1)
        s0 = s1 = self.bigDAG3(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits, aux_logits
    
    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p



class EvaluateRelaxedStage(SearchStage):
    """
    DAG for search
    Each edge is mixed and continuous relaxed
    """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, genotype, n_big_nodes, stem_multiplier=4, cell_multiplier=4, spec_cell=False, slide_window=3, auxiliary=True):
        """
        C_in: # of input channels
        C: # of starting model channels
        n_classes: # of classes
        n_layers: # of layers
        n_big_nodes: # of intermediate n_cells  # 6
        genotype: the shape of normal cell and reduce cell
        """
        super().__init__(input_size, C_in, C, n_classes, n_layers, genotype, n_big_nodes, stem_multiplier=stem_multiplier, cell_multiplier=cell_multiplier, spec_cell=spec_cell, slide_window=slide_window)
      
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1 

        C_pp, C_p, C_cur = cell_multiplier * C, cell_multiplier * C, C
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            if i in range(n_layers // 3):
                reduction = False
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                self.cells.append(cell)
            if i in [n_layers // 3]:
                self.bigDAG1 = SearchBigDAG(n_big_nodes, self.cells, 0, n_layers // 3, stem_multiplier*C_cur, stem_multiplier*C_cur, cell_multiplier * C_cur, window=self.window)

                reduction = True
                C_pp = C_p = 2*cell.multiplier*C_cur
                C_cur *= 2
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                self.cells.append(cell)
            if i in range(n_layers // 3 + 1, 2 * n_layers // 3):
                reduction = False
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                self.cells.append(cell)
            if i in [2 * n_layers // 3]:
                self.bigDAG2 = SearchBigDAG(n_big_nodes, self.cells, n_layers // 3 + 1, 2 * n_layers // 3, C_pp, C_p, cell_multiplier * C_cur, window=self.window)

                reduction = True
                C_pp = C_p = 2*cell.multiplier*C_cur
                C_cur *= 2
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                self.cells.append(cell)
            if i in range(2 * n_layers // 3 + 1, n_layers):
                reduction = False
                cell = GetCell(genotype, C_pp, C_p, C_cur, reduction) if not spec_cell else Get_StageSpecified_Cell(genotype, C_pp, C_p, C_cur, False, reduction, i, n_layers)
                self.cells.append(cell)
            if i == self.aux_pos:
                if input_size == setting.IMAGENET_SIZE:
                    self.aux_head = AuxiliaryHeadImagenet(14, C_p, n_classes)
                else:
                    self.aux_head = AuxiliaryHead(input_size // 4, C_p, n_classes)


            C_pp, C_p = cell_multiplier*C_cur, cell_multiplier*C_cur

        # self.bigDAG1 = SearchBigDAG(n_big_nodes, self.cells, 0, n_layers // 3, 4 * C)
        # self.bigDAG2 = SearchBigDAG(n_big_nodes, self.cells, n_layers // 3 + 1, 2 * n_layers // 3, 8 * C)
        self.bigDAG3 = SearchBigDAG(n_big_nodes, self.cells, 2 * n_layers // 3 + 1, n_layers, C_pp, C_p, cell_multiplier * C_cur, window=self.window)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32 * C, n_classes)
    
    def forward(self, x, weights_DAG):
        s0 = s1 = self.stem(x)
        s0 = s1 = self.bigDAG1(s0, s1, weights_DAG[0 * self.n_big_nodes: 1 * self.n_big_nodes])
        s0 = s1 = self.cells[1 * self.n_layers // 3](s0, s1)
        s0 = s1 = self.bigDAG2(s0, s1, weights_DAG[1 * self.n_big_nodes: 2 * self.n_big_nodes])
        s0 = s1 = self.cells[2 * self.n_layers // 3](s0, s1)
        aux_logits = self.aux_head(s1)
        s0 = s1 = self.bigDAG3(s0, s1, weights_DAG[2 * self.n_big_nodes: 3 * self.n_big_nodes])

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits, aux_logits


class EvaluateRelaxedStageController(SearchStageController):
    """" Augmented DAG-CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, criterion, genotype, stem_multiplier=4, device_ids=None, spec_cell=False, slide_window=3, auxiliary=False):

        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
            genotype: the struct of normal cell & reduce cell
            DAG: the struct of big-DAG
        """
        super().__init__(input_size, C_in, C, n_classes, n_layers, criterion, genotype, stem_multiplier=stem_multiplier, device_ids=device_ids, spec_cell=spec_cell, slide_window=slide_window)
       
        self.net = EvaluateRelaxedStage(input_size, C_in, C, n_classes, n_layers, genotype, self.n_big_nodes, stem_multiplier=stem_multiplier, spec_cell=spec_cell, slide_window=self.window, auxiliary=auxiliary)
    
    def print_alphas(self, logger):
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))
        
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - DAG")
        for alpha in self.alpha_DAG:
            logger.info(alpha)
        logger.info("#####################")

        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def forward(self, x):
        weights_DAG = [alpha for alpha in self.alpha_DAG]

        if len(self.device_ids) == 1:
            return self.net(x, weights_DAG)
        
        xs = nn.parallel.scatter(x, self.device_ids)
        wDAG_copies = broadcast_list(weights_DAG, self.device_ids)
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wDAG_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def drop_path_prob(self, p):
        for module in self.net.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
