from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype2 = namedtuple('Genotype2', 'DAG1 DAG1_concat DAG2 DAG2_concat DAG3 DAG3_concat')
Genotype3 = namedtuple('Genotype3', 'normal1 normal1_concat reduce1 reduce1_concat normal2 normal2_concat reduce2 reduce2_concat normal3 normal3_concat')
