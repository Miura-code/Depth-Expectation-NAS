# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)



""" Cells for architecture search """
from collections import OrderedDict
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

import genotypes.genotypes as gt
from models.search_stage import broadcast_list
from models.search_cell import SearchCell, SearchCellPartialConnection
from utils import setting


class SearchCellCNN(nn.Module):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_nodes = n_nodes

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
            
            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
    
    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            if i < self.n_layers // 3:
                weights = weights_normal[0: self.n_nodes]
            elif i == self.n_layers // 3:
                weights = weights_reduce[0: self.n_nodes]
            elif i < 2 * self.n_layers // 3:
                weights = weights_normal[self.n_nodes: 2 * self.n_nodes]
            elif i == 2 * self.n_layers // 3:
                weights = weights_reduce[self.n_nodes: 2 * self.n_nodes]
            elif i > 2 * self.n_layers // 3:
                weights = weights_normal[2 * self.n_nodes: 3 * self.n_nodes]
            else:
                raise EOFError

            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits


class SearchCellController(nn.Module):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=4, device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for _ in range(3):
            for i in range(n_nodes):
                self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
        for _ in range(2):
            for i in range(n_nodes):
                self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))


        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        
        self.net = SearchCellCNN(input_size, C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)
    
    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce)
        
        xs = nn.parallel.scatter(x, self.device_ids)
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def loss(self, X, guide_y, y, return_detail=False):
        logits = self.forward(X)
        return self.criterion(logits, guide_y, y, return_detail=return_detail)
    
    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
        
    def genotype(self):
        # gene_normal = gt.parse(self.alpha_normal, k=2)
        # gene_reduce = gt.parse(self.alpha_reduce, k=2)
        gene_normal1 = gt.parse_c(self.alpha_normal[0 * self.n_nodes: 1 * self.n_nodes], k=2)
        gene_reduce1 = gt.parse_c(self.alpha_reduce[0 * self.n_nodes: 1 * self.n_nodes], k=2)
        gene_normal2 = gt.parse_c(self.alpha_normal[1 * self.n_nodes: 2 * self.n_nodes], k=2)
        gene_reduce2 = gt.parse_c(self.alpha_reduce[1 * self.n_nodes: 2 * self.n_nodes], k=2)
        gene_normal3 = gt.parse_c(self.alpha_normal[2 * self.n_nodes: 3 * self.n_nodes], k=2)
        
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype3(normal1=gene_normal1, normal1_concat=concat,
                            reduce1=gene_reduce1, reduce1_concat=concat,
                            normal2=gene_normal2, normal2_concat=concat,
                            reduce2=gene_reduce2, reduce2_concat=concat,
                            normal3=gene_normal3, normal3_concat=concat)
    
    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

class SearchCellCNN_PatiallyConnected(SearchCellCNN):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        super().__init__(input_size, C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

        C_cur = stem_multiplier * C
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            
            cell = SearchCellPartialConnection(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)
    
    def forward(self, x, weights_normal, weights_reduce, weights_beta):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            if i < self.n_layers // 3:
                weights = weights_normal[0: self.n_nodes]
                weights_b = weights_beta[0: self.n_nodes]
            elif i == self.n_layers // 3:
                weights = weights_reduce[0: self.n_nodes]
                weights_b = weights_beta[self.n_nodes: 2 * self.n_nodes]
            elif i < 2 * self.n_layers // 3:
                weights = weights_normal[self.n_nodes: 2 * self.n_nodes]
                weights_b = weights_beta[2 * self.n_nodes: 3 * self.n_nodes]
            elif i == 2 * self.n_layers // 3:
                weights = weights_reduce[self.n_nodes: 2 * self.n_nodes]
                weights_b = weights_beta[3 * self.n_nodes: 4 * self.n_nodes]
            elif i > 2 * self.n_layers // 3:
                weights = weights_normal[2 * self.n_nodes: 3 * self.n_nodes]
                weights_b = weights_beta[4 * self.n_nodes: 5 * self.n_nodes]
            else:
                raise EOFError
            

            s0, s1 = s1, cell(s0, s1, weights, weights_b)
        
        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return logits

class SearchCellController_PartiallyConntected(SearchCellController):
    def __init__(self, input_size, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=4, device_ids=None):
        super().__init__(input_size, C_in, C, n_classes, n_layers, criterion, n_nodes, stem_multiplier, device_ids)

        self.beta = nn.ParameterList()

        for _ in range(5):
            for i in range(n_nodes):
                self.beta.append(nn.Parameter(1e-3 * torch.randn(i + 2)))

        self._betas = []
        for n, p in self.named_parameters():
            if 'beta' in n:
                self._betas.append((n, p))

        self.net = SearchCellCNN_PatiallyConnected(input_size, C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]
        weights_beta = [F.softmax(beta, dim=-1) for beta in self.beta]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce, weights_beta)
        
        xs = nn.parallel.scatter(x, self.device_ids)
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)
        wbeta_copies = broadcast_list(weights_beta, self.device_ids)
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies, wbeta_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])
    
    def genotype(self):
        # gene_normal = gt.parse(self.alpha_normal, k=2)
        # gene_reduce = gt.parse(self.alpha_reduce, k=2)
        gene_normal1 = gt.parse_c_edgeNormalization(self.alpha_normal[0 * self.n_nodes: 1 * self.n_nodes], self.beta[0 * self.n_nodes: 1 * self.n_nodes], k=2)
        gene_reduce1 = gt.parse_c_edgeNormalization(self.alpha_reduce[0 * self.n_nodes: 1 * self.n_nodes], self.beta[1 * self.n_nodes: 2 * self.n_nodes], k=2)
        gene_normal2 = gt.parse_c_edgeNormalization(self.alpha_normal[1 * self.n_nodes: 2 * self.n_nodes], self.beta[2 * self.n_nodes: 3 * self.n_nodes], k=2)
        gene_reduce2 = gt.parse_c_edgeNormalization(self.alpha_reduce[1 * self.n_nodes: 2 * self.n_nodes], self.beta[3 * self.n_nodes: 4 * self.n_nodes], k=2)
        gene_normal3 = gt.parse_c_edgeNormalization(self.alpha_normal[2 * self.n_nodes: 3 * self.n_nodes], self.beta[4 * self.n_nodes: 5 * self.n_nodes], k=2)
        
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype3(normal1=gene_normal1, normal1_concat=concat,
                            reduce1=gene_reduce1, reduce1_concat=concat,
                            normal2=gene_normal2, normal2_concat=concat,
                            reduce2=gene_reduce2, reduce2_concat=concat,
                            normal3=gene_normal3, normal3_concat=concat)

    def betas(self):
        for n, p in self._betas:
            yield p

    def named_betas(self):
        for n, p in self._betas:
            yield n, p
