# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)

""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""

from genotypes.__init__ import *

HS_DAS_CIFAR = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('max_pool_3x3', 2)], 
        [('max_pool_3x3', 2), ('skip_connect', 4)],
        [('max_pool_3x3', 3), ('skip_connect', 5)], 
        [('max_pool_3x3', 4), ('skip_connect', 6)]
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('max_pool_3x3', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('max_pool_3x3', 5)], 
        [('avg_pool_3x3', 4), ('skip_connect', 6)]
    ], 
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('max_pool_3x3', 2), ('skip_connect', 3)],
        [('max_pool_3x3', 3), ('skip_connect', 5)], 
        [('max_pool_3x3', 4), ('max_pool_3x3', 5)]
    ], 
    DAG3_concat=[6, 7]
)

HS_DAS_CIFAR_SKIP = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 5)]
    ], 
    DAG3_concat=[6, 7]
)

STAGE_HSDAS_V1 = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG3_concat=[6, 7]
)

STAGE_HSDAS_V2 = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG3_concat=[6, 7]
)

STAGE_HSDAS_V3 = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 5)]
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 5)]
    ], 
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 5)]
    ], 
    DAG3_concat=[6, 7]
)

STAGE_SHALLOW = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],  
        [('skip_connect', 0), ('skip_connect', 1)],
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],  
        [('skip_connect', 0), ('skip_connect', 1)],
    ],
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 0), ('skip_connect', 1)],  
        [('skip_connect', 0), ('skip_connect', 1)],
    ], 
    DAG3_concat=[6, 7]
)

STAGE_MIDDLE = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 3)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 5)],
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 3)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 5)],
    ],
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 3)], 
        [('skip_connect', 2), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 5)],
    ], 
    DAG3_concat=[6, 7]
)

STAGE_DEEP = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)],
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)],
    ],
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 5)], 
        [('skip_connect', 4), ('skip_connect', 6)],
    ], 
    DAG3_concat=[6, 7]
)

# Almost DARTS stage structure (Deepest structure)
STAGE_DARTS = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 4), ('skip_connect', 5)], 
        [('skip_connect', 5), ('skip_connect', 6)],
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 4), ('skip_connect', 5)], 
        [('skip_connect', 5), ('skip_connect', 6)],
    ],
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 4), ('skip_connect', 5)], 
        [('skip_connect', 5), ('skip_connect', 6)],
    ], 
    DAG3_concat=[6, 7]
)

STAGE_FULL_CASCADE = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 5)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6)],
    ], 
    DAG1_concat=[6, 7], 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 5)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6)],
    ], 
    DAG2_concat=[6, 7], 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 5)], 
        [('skip_connect', 0),('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 6)],
    ], 
    DAG3_concat=[6, 7]
)

NonDepth_BASELINE = Genotype2(
    DAG1=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 2), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 4)], 
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 6)]
        ], 
    DAG1_concat=range(6, 8), 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 1), ('skip_connect', 0)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 5)]
        ], 
    DAG2_concat=range(6, 8), 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 2)], 
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 5)]
        ], 
    DAG3_concat=range(6, 8))

BASELINE_BEST_STAGE = Genotype2(
    DAG1=[
        [('avg_pool_3x3', 1), ('avg_pool_3x3', 0)],
        [('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], 
        [('avg_pool_3x3', 3), ('avg_pool_3x3', 2)], 
        [('avg_pool_3x3', 4), ('avg_pool_3x3', 3)], 
        [('avg_pool_3x3', 5), ('avg_pool_3x3', 4)], 
        [('avg_pool_3x3', 6), ('avg_pool_3x3', 5)]
        ], 
    DAG1_concat=range(6, 8), 
    DAG2=[
        [('max_pool_3x3', 1), ('max_pool_3x3', 0)], 
        [('skip_connect', 2), ('max_pool_3x3', 1)],
        [('max_pool_3x3', 3), ('skip_connect', 2)], 
        [('max_pool_3x3', 4), ('skip_connect', 3)], 
        [('skip_connect', 5), ('avg_pool_3x3', 4)], 
        [('skip_connect', 6), ('avg_pool_3x3', 5)]
        ], 
    DAG2_concat=range(6, 8), 
    DAG3=[
        [('skip_connect', 1), ('skip_connect', 0)], 
        [('skip_connect', 2), ('skip_connect', 1)], 
        [('skip_connect', 3), ('skip_connect', 2)], 
        [('max_pool_3x3', 4), ('skip_connect', 3)], 
        [('skip_connect', 5), ('skip_connect', 4)], 
        [('max_pool_3x3', 6), ('skip_connect', 5)]
        ], 
    DAG3_concat=range(6, 8))

NonVArch_BASELINE = Genotype2(
    DAG1=[
        [('skip_connect', 1), ('skip_connect', 0)], 
        [('max_pool_3x3', 2), ('skip_connect', 1)], 
        [('avg_pool_3x3', 3), ('skip_connect', 2)], 
        [('max_pool_3x3', 4), ('max_pool_3x3', 3)], 
        [('max_pool_3x3', 5), ('skip_connect', 4)], 
        [('avg_pool_3x3', 6), ('skip_connect', 5)]
        ], 
    DAG1_concat=range(6, 8),
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)],
        [('avg_pool_3x3', 2), ('skip_connect', 1)],
        [('avg_pool_3x3', 3), ('skip_connect', 2)], 
        [('max_pool_3x3', 4), ('avg_pool_3x3', 3)], 
        [('avg_pool_3x3', 5), ('skip_connect', 4)], 
        [('avg_pool_3x3', 6), ('skip_connect', 5)]
        ], 
    DAG2_concat=range(6, 8), 
    DAG3=[
        [('skip_connect', 1), ('skip_connect', 0)], 
        [('skip_connect', 2), ('skip_connect', 1)], 
        [('skip_connect', 3), ('skip_connect', 2)], 
        [('skip_connect', 4), ('skip_connect', 3)], 
        [('skip_connect', 5), ('skip_connect', 4)], 
        [('skip_connect', 6), ('skip_connect', 5)]
        ], 
    DAG3_concat=range(6, 8))

NoDepthLoss_BEST = Genotype2(
    DAG1=[
        [('skip_connect', 1), ('skip_connect', 0)],
        [('skip_connect', 0), ('skip_connect', 1)],
        [('skip_connect', 1), ('skip_connect', 3)],
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 3), ('skip_connect', 4)],
        [('skip_connect', 4), ('skip_connect', 6)]
    ], 
    DAG1_concat=range(6, 8), 
    DAG2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 0)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 3), ('skip_connect', 2)],
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 5)]
        ], 
    DAG2_concat=range(6, 8), 
    DAG3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 2)], 
        [('skip_connect', 2), ('skip_connect', 3)], 
        [('skip_connect', 3), ('skip_connect', 4)], 
        [('skip_connect', 4), ('skip_connect', 5)]
        ], 
    DAG3_concat=range(6, 8)
)