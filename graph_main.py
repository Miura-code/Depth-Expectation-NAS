
import networkx
from config.graph_config import GraphConfig
from utils import visualize
from utils.graphs import *
from utils.visualize import plot2
import random
import numpy

import torch
import torch.nn as nn


def set_seed(seed=0):
    random.seed(seed)        # or any integer
    numpy.random.seed(seed)

def main(config):
    set_seed(config.seed)
    GraphUtils = Graph_Utilities()
    
    # dag = config.DAG
    # plot2(dag.DAG1, './graphs/DAG1', config.DAG_name)
    # plot2(dag.DAG2, './graphs/DAG2', config.DAG_name)
    # plot2(dag.DAG3, './graphs/DAG3', config.DAG_name)
    
    # g1 = GraphUtils.create_StageGraph(dag.DAG1)
    # g2 = GraphUtils.create_StageGraph(dag.DAG2)
    # g3 = GraphUtils.create_StageGraph(dag.DAG3)
    # GraphUtils.visualize_graphs([g1, g2, g3], "./graphs/{}".format(config.DAG_name))
    # distance = GraphUtils.graphEditDistance(g1, g2)

    alpha_DAG = []
    for _ in range(3):
        for i in range(6):
            alpha_DAG.append(torch.tensor(1e-3 * torch.randn(i + 2, len(gt.PRIMITIVES3))))

    print(alpha_DAG[:6])
    dag = GraphUtils.alpha_to_DAG(alpha_DAG)
    graph_nxs = GraphUtils.make_NX_Graph(dag)
    print(dag)
    print(graph_nxs)
    input()
    GraphUtils.visualize_graphs(graph_nxs, "./graphs/sample_dag")

    L1 = GraphUtils.Laplacian_from_genotype(alpha_DAG[:6])
    L2 = GraphUtils.Laplacian_from_genotype(alpha_DAG[:6])

    value1, vector1 = GraphUtils.cal_eigen(L1)
    value2, vector2 = GraphUtils.cal_eigen(L2)

    similarity = GraphUtils.cal_cosine_similarity(value1, value2)
    print(similarity)


        
    pass

if __name__ == "__main__":
    config = GraphConfig()
    main(config)
