""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype2 = namedtuple('Genotype2', 'DAG1 DAG1_concat DAG2 DAG2_concat DAG3 DAG3_concat')
Genotype3 = namedtuple('Genotype3', 'normal1 normal1_concat reduce1 reduce1_concat normal2 normal2_concat reduce2 reduce2_concat normal3 normal3_concat')


DARTS_V1 = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], 
        [('skip_connect', 0), ('sep_conv_3x3', 1)], 
        [('skip_connect', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 0), ('skip_connect', 2)]
    ], 
    normal_concat=[2, 3, 4, 5], 
    reduce=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)], 
        [('skip_connect', 2), ('max_pool_3x3', 0)], 
        [('max_pool_3x3', 0), ('skip_connect', 2)], 
        [('skip_connect', 2), ('avg_pool_3x3', 0)]
    ], 
    reduce_concat=[2, 3, 4, 5]
)
DARTS_V2 = Genotype(
    normal=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 1), ('skip_connect', 0)], 
        [('skip_connect', 0), ('dil_conv_3x3', 2)]
    ], 
    normal_concat=[2, 3, 4, 5], 
    reduce=[
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)], 
        [('skip_connect', 2), ('max_pool_3x3', 1)], 
        [('max_pool_3x3', 0), ('skip_connect', 2)], 
        [('skip_connect', 2), ('max_pool_3x3', 1)]
        ], 
        reduce_concat=[2, 3, 4, 5]
    )

PC_DARTS_CIFAR = Genotype(
    normal=[
        [('sep_conv_3x3', 1), ('skip_connect', 0)], 
        [('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], 
        [('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], 
        [('avg_pool_3x3', 0), ('dil_conv_3x3', 1)]
    ],
    normal_concat=range(2, 6), 
    reduce=[
        [('sep_conv_5x5', 1), ('max_pool_3x3', 0)], 
        [('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], 
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)]
    ], 
    reduce_concat=range(2, 6)
)

HC_DAS = Genotype3(
    normal1=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 2), ('sep_conv_3x3', 3)]
    ],
    normal1_concat=range(2, 6), 
    reduce1=[
        [('max_pool_3x3', 0), ('skip_connect', 0)], 
        [('max_pool_3x3', 0), ('max_pool_3x3', 2)], 
        [('sep_conv_3x3', 0), ('skip_connect', 1)], 
        [('sep_conv_3x3', 0), ('dil_conv_3x3', 4)]
    ], 
    reduce1_concat=range(2, 6),
    normal2=[
        [('skip_connect', 0), ('dil_conv_3x3', 1)], 
        [('skip_connect', 0), ('sep_conv_3x3', 1)], 
        [('skip_connect', 0), ('sep_conv_3x3', 1)], 
        [('skip_connect', 0), ('sep_conv_3x3', 1)]
    ],
    normal2_concat=range(2, 6), 
    reduce2=[
        [('dil_conv_3x3', 0), ('max_pool_3x3', 1)], 
        [('max_pool_3x3', 1), ('dil_conv_5x5', 2)], 
        [('max_pool_3x3', 1), ('dil_conv_5x5', 3)], 
        [('max_pool_3x3', 1), ('skip_connect', 4)]
    ], 
    reduce2_concat=range(2, 6),
    normal3=[
        [('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)], 
        [('dil_conv_3x3', 0), ('sep_conv_5x5', 2)], 
        [('sep_conv_5x5', 3), ('sep_conv_3x3', 4)]
    ],
    normal3_concat=range(2, 6)
)

SIMPLE_V1 = Genotype(
    normal=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('none', 1), ('avg_pool_3x3', 2)], 
        [('skip_connect', 2), ('sep_conv_5x5', 3)], 
        [('none', 3), ('avg_pool_3x3', 4)]
    ],
    normal_concat=range(2, 6), 
    reduce=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('none', 1), ('avg_pool_3x3', 2)], 
        [('skip_connect', 2), ('sep_conv_5x5', 3)], 
        [('none', 3), ('avg_pool_3x3', 4)]
    ], 
    reduce_concat=range(2, 6)
)

PC_DARTS_CIFAR = Genotype(
    normal=[
        [('sep_conv_3x3', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('dil_conv_3x3', 2)], 
        [('sep_conv_3x3', 1), ('dil_conv_3x3', 3)], 
        [('sep_conv_3x3', 1), ('sep_conv_5x5', 4)]
    ],
    normal_concat=range(2, 6), 
    reduce=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('none', 1), ('avg_pool_3x3', 2)], 
        [('skip_connect', 2), ('sep_conv_5x5', 3)], 
        [('none', 3), ('avg_pool_3x3', 4)]
    ], 
    reduce_concat=range(2, 6)
)

PDARTS = Genotype(
    normal=[
        [('skip_connect', 0), ('dil_conv_3x3', 1)],
        [('skip_connect', 0), ('sep_conv_3x3', 1)],
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 3)],
        [('sep_conv_3x3',0), ('dil_conv_5x5', 4)]
    ], 
    normal_concat=range(2, 6), 
    reduce=[
        [('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], 
        [('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], 
        [('max_pool_3x3', 0), ('dil_conv_3x3', 1)], 
        [('dil_conv_3x3', 1), ('dil_conv_5x5', 3)]
    ], 
    reduce_concat=range(2, 6)
)

BASELINE1 = Genotype3(
    normal1=[
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], 
        [('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 4)]
    ], 
    normal1_concat=range(2, 6), 
    reduce1=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3',1)], 
        [('skip_connect', 1), ('avg_pool_3x3', 0)], 
        [('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], 
        [('avg_pool_3x3', 0), ('skip_connect', 3)]
    ], 
    reduce1_concat=range(2, 6), 
    normal2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 1), ('skip_connect', 0)], 
        [('skip_connect', 0), ('sep_conv_3x3', 2)], 
        [('skip_connect', 0), ('skip_connect', 1)]
    ], 
    normal2_concat=range(2, 6), 
    reduce2=[
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], 
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], 
        [('avg_pool_3x3', 0), ('skip_connect', 2)], 
        [('avg_pool_3x3', 0), ('skip_connect', 2)]
    ], 
    reduce2_concat=range(2, 6), 
    normal3=[
        [('skip_connect', 0), ('dil_conv_3x3', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('skip_connect', 2)]
    ], normal3_concat=range(2, 6))

BASELINE2 = Genotype3(
    normal1=[
        [('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], 
        [('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], 
        [('sep_conv_3x3', 0), ('max_pool_3x3', 1)], 
        [('sep_conv_5x5', 1), ('dil_conv_3x3', 4)]
    ], 
    normal1_concat=range(2, 6), 
    reduce1=[
        [('dil_conv_5x5', 1), ('dil_conv_3x3',0)], 
        [('dil_conv_3x3', 1), ('max_pool_3x3', 0)], 
        [('dil_conv_5x5', 1), ('dil_conv_5x5', 3)], 
        [('dil_conv_5x5', 3), ('dil_conv_3x3', 2)]
    ], 
    reduce1_concat=range(2, 6), 
    normal2=[
        [('dil_conv_3x3', 1), ('sep_conv_5x5', 0)], 
        [('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], 
        [('sep_conv_5x5', 0), ('avg_pool_3x3', 1)], 
        [('sep_conv_3x3', 2), ('sep_conv_5x5', 1)]
    ], 
    normal2_concat=range(2, 6), 
    reduce2=[
        [('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], 
        [('sep_conv_3x3', 2), ('sep_conv_5x5', 1)], 
        [('skip_connect', 0), ('sep_conv_5x5', 1)], 
        [('sep_conv_3x3', 3), ('dil_conv_5x5', 1)]
    ], 
    reduce2_concat=range(2, 6), 
    normal3=[
        [('sep_conv_5x5', 0), ('dil_conv_5x5', 1)], 
        [('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], 
        [('dil_conv_5x5', 2), ('sep_conv_3x3', 1)], 
        [('sep_conv_5x5', 4), ('sep_conv_5x5', 2)]
    ], 
    normal3_concat=range(2, 6)
) 

BASELINE224 = Genotype3(
    normal1=[
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], 
        [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)], 
        [('sep_conv_3x3', 1), ('sep_conv_3x3', 3)]
    ], 
    normal1_concat=range(2, 6), 
    reduce1=[
        [('skip_connect', 1), ('sep_conv_5x5', 0)], 
        [('sep_conv_3x3', 2), ('sep_conv_5x5', 0)], 
        [('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], 
        [('max_pool_3x3', 0), ('max_pool_3x3', 1)]
    ], 
    reduce1_concat=range(2, 6), 
    normal2=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('dil_conv_3x3', 2)], 
        [('skip_connect', 0), ('skip_connect', 2)], 
        [('skip_connect', 0), ('skip_connect', 1)]
    ], 
    normal2_concat=range(2, 6), 
    reduce2=[
        [('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], 
        [('avg_pool_3x3', 0), ('skip_connect', 2)], 
        [('avg_pool_3x3', 0), ('skip_connect', 2)], 
        [('skip_connect', 2), ('avg_pool_3x3', 0)]
    ], reduce2_concat=range(2, 6), 
    normal3=[
        [('skip_connect', 0), ('skip_connect', 1)], 
        [('skip_connect', 0), ('dil_conv_5x5', 1)], 
        [('skip_connect', 0), ('dil_conv_3x3', 1)], 
        [('skip_connect', 0), ('skip_connect', 1)]
    ], 
    normal3_concat=range(2, 6)
)