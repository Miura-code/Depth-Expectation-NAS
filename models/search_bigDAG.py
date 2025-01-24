# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)



import torch
import torch.nn as nn
from models import ops


class SearchBigDAG(nn.Module):
    def __init__(self, n_big_nodes, cells, start_p, end_p, C_pp, C_p, C, window=3):
        super().__init__()
        self.n_big_nodes = n_big_nodes
        self.window = window
        self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        self.DAG = nn.ModuleList()
        for i in range(self.n_big_nodes):
            self.DAG.append(nn.ModuleList())
            if i + 2 < self.window:
                for _ in range(2 + i):
                    stride = 1
                    op = ops.MixedOp(C, stride)
                    self.DAG[i].append(op)
            else:
                for _ in range(self.window):
                    stride = 1
                    op = ops.MixedOp(C, stride)
                    self.DAG[i].append(op)
        
        for k in range(start_p, end_p):
            self.DAG.append(cells[k])
    
    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for j, (edges, w_list) in enumerate(zip(self.DAG, w_dag)):
            if j + 2 < self.window:
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))
            else:
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states[-self.window:], w_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))

        s_out = torch.cat(states[self.n_big_nodes:], dim=1)
        return s_out
    
class SearchBigDAGPartiallyConnection(nn.Module):
    def __init__(self, n_big_nodes, cells, start_p, end_p, C_pp, C_p, C, window=3):
        super().__init__()
        self.n_big_nodes = n_big_nodes
        self.window = window
        self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        self.DAG = nn.ModuleList()
        for i in range(self.n_big_nodes):
            self.DAG.append(nn.ModuleList())
            if i + 2 < self.window:
                for _ in range(2 + i):
                    stride = 1
                    op = ops.MixedOpPC(C, stride)
                    self.DAG[i].append(op)
            else:
                for _ in range(self.window):
                    stride = 1
                    op = ops.MixedOpPC(C, stride)
                    self.DAG[i].append(op)
        
        for k in range(start_p, end_p):
            self.DAG.append(cells[k])
    
    def forward(self, s0, s1, w_dag, bw_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for j, (edges, w_list, bw_list) in enumerate(zip(self.DAG, w_dag, bw_dag)):
            if j + 2 < self.window:
                s_cur = sum(b * edges[i](s, w) for i, (s, w, b) in enumerate(zip(states, w_list, bw_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))
            else:
                s_cur = sum(b * edges[i](s, w) for i, (s, w, b) in enumerate(zip(states[-self.window:], w_list, bw_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))

        s_out = torch.cat(states[self.n_big_nodes:], dim=1)
        return s_out

class SearchBigDAG_CS(SearchBigDAG):
    def __init__(self, n_big_nodes, cells, start_p, end_p, C_pp, C_p, C, window=3):
        super().__init__(n_big_nodes, cells, start_p, end_p, C_pp, C_p, C, window=window)
    
    def forward(self, s0, s1, w_dag, w_concat):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for j, (edges, w_list) in enumerate(zip(self.DAG, w_dag)):
            if j + 2 < self.window:
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))
            else:
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states[-self.window:], w_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))
        
        s_out = list()
        for i in range(4, self.n_big_nodes + 1):
            s_out.append(torch.cat(states[i: i + 2], dim=1))
        
        ss = sum([wc * so for (wc, so) in zip(w_concat, s_out)])
        return ss

class SearchBigDAG_BETA(SearchBigDAG):
    def __init__(self, n_big_nodes, cells, start_p, end_p, C_pp, C_p, C, window=3):
        super().__init__(n_big_nodes, cells, start_p, end_p, C_pp, C_p, C, window=window)
    
    def forward(self, s0, s1, w_dag, w_concat):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for j, (edges, w_list) in enumerate(zip(self.DAG, w_dag)):
            if j + 2 < self.window:
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))
            else:
                s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states[-self.window:], w_list)))
                states.append(self.DAG[j + self.n_big_nodes](s_cur, s_cur))
        
        s_out = list()
        for i in range(2, self.n_big_nodes + 1):
            for j in range(i+1, self.n_big_nodes + 2):
                s_out.append(torch.cat([states[i], states[j]], dim=1))
        
        ss = sum([wc * so for (wc, so) in zip(w_concat, s_out)])
        return ss
