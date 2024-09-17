
import networkx
from config.graph_config import GraphConfig
from genotypes.genotype_stage import *
from genotypes.genotypes import parse_dag_to_alpha
import teacher_models
from utils import visualize
from utils.graphs import *
from utils.graphs import create_StageGraph, graphEditDistance, visualize_graphs
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
    
    dag = config.DAG
    # plot2(dag.DAG1, './graphs/DAG1', config.DAG_name)
    # plot2(dag.DAG2, './graphs/DAG2', config.DAG_name)
    # plot2(dag.DAG3, './graphs/DAG3', config.DAG_name)
    
    # g1 = GraphUtils.create_StageGraph(dag.DAG1)
    # g2 = GraphUtils.create_StageGraph(dag.DAG2)
    # g3 = GraphUtils.create_StageGraph(dag.DAG3)
    # GraphUtils.visualize_graphs([g1, g2, g3], "./graphs/{}".format(config.DAG_name))
    # distance = GraphUtils.graphEditDistance(g1, g2)

    # ---------- 接続行列or隣接行列 ----------
    # alpha_DAG = []
    # for _ in range(3):
    #     for i in range(6):
    #         alpha_DAG.append(torch.tensor(1e-3 * torch.randn(i + 2, len(gt.PRIMITIVES3))))
    # print(alpha_DAG[:6])
    # dag = GraphUtils.alpha_to_DAG(alpha_DAG)
    # graph_nxs = GraphUtils.make_NX_Graph(dag)
    # print(dag)
    # print(graph_nxs)
    # input()
    # GraphUtils.visualize_graphs(graph_nxs, "./graphs/sample_dag")

    # L1 = GraphUtils.Laplacian_from_genotype(alpha_DAG[:6], True)
    # L2 = GraphUtils.Laplacian_from_genotype(alpha_DAG[:6], False)
    # print("接続行列からラプラシアン", L1)
    # print("隣接行列からラプラシアン", L2)
    # print("L1 - L2", L1 - L2)
    # input()
    # # L2 = GraphUtils.Laplacian_from_genotype(alpha_DAG[:6])

    # value1, vector1 = GraphUtils.cal_eigen(L1)
    # value2, vector2 = GraphUtils.cal_eigen(L2)

    # print("接続行列からラプラシアンの固有値\n", value1)
    # print("隣接行列からラプラシアンの固有値\n", value2)

    # ---------- 類似度計測 ----------
    # model = teacher_models.__dict__["h_das_224baseline"](num_classes = 100)
    # setattr(model, "alpha_DAG", parse_dag_to_alpha(model.DAG, 6, n_ops=4, window=15, device="cpu"))
    model_alpha_DAG = parse_dag_to_alpha(STAGE_SHALLOW, 6, n_ops=4, window=15, device="cpu")
    alpha_DAG = []
    for _ in range(3):
        for i in range(6):
            alpha_DAG.append(torch.tensor(1e-3 * torch.randn(i + 2, len(gt.PRIMITIVES3))))
    L1 = GraphUtils.Laplacian_from_genotype(alpha_DAG[:6], False)
    value1, vector1 = GraphUtils.cal_eigen(L1)
    L2 = GraphUtils.Laplacian_from_genotype(model_alpha_DAG[:6], False)
    value2, vector2 = GraphUtils.cal_eigen(L2)
    print(value1.real, value2.real)
    similarity = GraphUtils.cal_cosine_similarity(value1.real, value2.real)
    print(similarity)

    DAG1 = GraphUtils.alpha_to_DAG(alpha_DAG)
    DAG2 = GraphUtils.alpha_to_DAG(model_alpha_DAG)

    plot_path = "./graphs/"
    plot2(DAG1.DAG1, plot_path + 'random-DAG', "TEST")
    plot2(DAG2.DAG1, plot_path + 'teacher-DAG', "TEST")
    
    GraphUtils.visualize_graphs(GraphUtils.make_NX_Graph(DAG1), plot_path + 'random-DAG-NX')
    GraphUtils.visualize_graphs(GraphUtils.make_NX_Graph(DAG2), plot_path + 'teacher-DAG-NX')
    
    # 類似計算
    # _DAG = parse_dag_to_alpha(STAGE_SHALLOW, 6, n_ops=4, window=15, device="cpu")
    # alpha_DAG = []
    # for _ in range(3):
    #     for i in range(6):
    #         alpha_DAG.append(torch.tensor(1e-3 * torch.randn(i + 2, len(gt.PRIMITIVES3))))
    # L1 = GraphUtils.Laplacian_from_genotype(alpha_DAG, False)
    # print(L1)
    # value1, vector1 = GraphUtils.cal_eigen(L1)
    # print(value1)

    g1 = create_StageGraph(dag.DAG1)
    g2 = create_StageGraph(dag.DAG2)
    g3 = create_StageGraph(dag.DAG3)
    visualize_graphs([g1, g2, g3], "./graphs/{}".format(config.DAG_name))
    distance = graphEditDistance(g1, g2)
    
    pass

if __name__ == "__main__":
    config = GraphConfig()
    main(config)
