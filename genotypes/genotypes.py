# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
import pickle
import torch
import torch.nn as nn
from models import ops

from genotypes.__init__ import *
from genotypes.genotype_cell import *
from genotypes.genotype_stage import *

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
]

PRIMITIVES2 = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'none'
]

PRIMITIVES3 = [
    'skip_connect', # identity
    'none'
]


def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity): # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    if s.endswith(".pickle"):
        genotype = load_DAG(s)
    else:
        genotype = eval(s)

    return genotype


def parse(alpha, k, window=3):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES2[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for i, edges in enumerate(alpha):
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES2[prim_idx]
            if i + 2 < window:
                node_gene.append((prim, edge_idx.item()))
            else:
                node_gene.append((prim, edge_idx.item() + (i + 2 - window)))

        gene.append(node_gene)

    return gene

def parse_fullcascade(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES2[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for i, edges in enumerate(alpha):
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES2[prim_idx]
            node_gene.append((prim, edge_idx.item()))
            
        gene.append(node_gene)

    return gene

def parse_edgeNormalization(alpha, beta, k, window=3):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES2[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge) 
    # (partially connectionの場合はedge normalizationのパラメータbetaを掛けた値をedge scoreとする)
    for i, (edges, b_edges) in enumerate(zip(alpha, beta)):
        # edges: Tensor(n_edges, n_ops)
        W = torch.ones_like(edges)
        for j in range(edges.shape[0]):
            W[j,:] = edges[j,:] * b_edges[j]
        edge_max, primitive_indices = torch.topk(W[:, :-1], 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES2[prim_idx]
            if i + 2 < window:
                node_gene.append((prim, edge_idx.item()))
            else:
                node_gene.append((prim, edge_idx.item() + (i + 2 - window)))

        gene.append(node_gene)

    return gene


def parse_c(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for i, edges in enumerate(alpha):
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene

def parse_c_edgeNormalization(alpha, beta, k):
    """
    parse continuous beta(params for edge normalization by PCDARTS) to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for i, (edges, b_edges) in enumerate(zip(alpha, beta)):
        # edges: Tensor(n_edges, n_ops)
        W = torch.ones_like(edges)
        for j in range(edges.shape[0]):
            W[j,:] = edges[j,:] * b_edges[j]
        edge_max, primitive_indices = torch.topk(W[:, :-1], 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


def parse_concat(beta):
    """
    parse continuous beta ti discrete concat
    beta is ParameterList:
    ParameterList [
        Parameter(1, 1),
        Parameter(1, 1),
        ...
    ]

    concat is list:
    range(2, 4)
    range(5, 7)
    ...
    range(6, 8)
    """
    _, index = torch.topk(beta, 1, dim=0)
    return range(index + 4, index + 6)

def save_DAG(DAG, path, is_best=False):
    if is_best:
        path = path + '-best'
    with open(path + '.pickle', mode='wb') as f:
        pickle.dump(DAG, f)
        
def load_DAG(path):
    with open(path, 'rb') as rh:
        dag = pickle.load(rh)
        
    return dag

def parse_dag_to_alpha(dag, n_big_nodes, n_ops=len(PRIMITIVES3), K=2, window=3, device='cpu'):
    alpha = []
    dags = [dag.DAG1, dag.DAG2, dag.DAG3]
    for j, DAG in enumerate(dags):
        for i in range(len(DAG)):
            # sliding window
            if i + 2 < window:
                alpha.append(torch.zeros(i + 2, n_ops, device=device))
            else:
                alpha.append(torch.zeros(window, n_ops, device=device))
            for k in range(K):
                alpha[-1][DAG[i][k][1]][PRIMITIVES2.index(DAG[i][k][0])] = torch.tensor(1.0)
    return alpha
            
    