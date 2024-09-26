from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import genotypes.genotypes as gt
from utils import visualize

class Graph_Utilities():
    def __init__(self):
        # self.num_ops = len(gt.PRIMITIVES2)
        self.num_ops = len(gt.PRIMITIVES3)
        self.window = 15

        self.n_nodes = 8
        self.n_big_dag = self.n_nodes - 2
        num_edges = 0
        for i in range(2, self.n_nodes - 1):
            if i + 2 < self.window:
                num_edges += self.num_ops * (i)
            else:
                num_edges += self.num_ops * (self.window)
        self.n_edges = num_edges

        pass

    def graphEditDistance(self, g1, g2):
        """Get the distance between two graphs .

        Args:
            g1 ([type]): [description]
            g2 ([type]): [description]

        Returns:
            [type]: [description]
        """
        # todo グラフの編集距離
        distance = nx.graph_edit_distance(g1, g2)

        return distance

    def _make_StageGraph(self, dag):
        """
        nwtworkXを使ったグラフを作成する
        Args:
            genotype: グラフ情報
        Return:
            G
        """
        G = nx.DiGraph()
        G.add_nodes_from([0, 1])
        for k, edges in enumerate(dag):
            G.add_node(k+2)
            G.add_edges_from([(edges[0][1], k+2, {"ops": edges[0][0]}), (edges[1][1], k+2, {"ops": edges[1][0]})])
        
        return G
    
    def make_NX_Graph(self, DAG):
        """Return a list of NX graph representing the DAG .

        Args:
            DAG ([type]): [description]

        Returns:
            [type]: [description]
        """
        G1 = self._make_StageGraph(DAG.DAG1)
        G2 = self._make_StageGraph(DAG.DAG2)
        G3 = self._make_StageGraph(DAG.DAG3)

        return G1, G2, G3


    def visualize_graphs(self, graphs, path, graphviz=False, DAG=None):
        """Visualize the graph of the networkx graph .

        Args:
            graphs ([type]): [description]
            path ([type]): [description]
            graphviz (bool, optional): [description]. Defaults to False.
            DAG ([type], optional): [description]. Defaults to None.
        """
        
            
        num_graphs = len(graphs)
        fig = plt.figure()
        for i in range(num_graphs):
            subax = plt.subplot(1, num_graphs, i+1, title="DAG"+str(i+1))
            nx.draw_networkx(graphs[i], with_labels=True, ax=subax)
            
        plt.savefig(path + '.png')

    def Laplacian_from_genotype(self, alpha_dag, C=True):
        """構造パラメータからグラフラプラシアン行列を作成する

        Args:
            alpha_dag ([type]): 構造パラメータ（１ステージ分）

        Returns:
            torch.tensor: グラフラプラシアン
        """
        if C:
            B = self._make_connection_matrix(alpha_dag)

            W = []
            e = 0
            for alphas_node in alpha_dag:
                alphas_node = torch.softmax(alphas_node, dim=-1)
                for alpha in alphas_node:
                    # W.append(alpha[:1])
                    # print(W)
                    W.append(torch.tensor([1.]))
            W = torch.diag(torch.concat(W))

            L = torch.matmul(torch.matmul(B, W), torch.t(B))
        else:
            A = self._make_neighber_matrix(alpha_dag)
            # A = self._make_neighber_matrix2(alpha_dag)

            # print(A)
            D = torch.zeros_like(A)
            for i in range(self.n_nodes):
                D[i, i] = torch.sum(A[i])
            
            L = D - A

            # I = torch.eye(A.size()[0])
            # D_ = torch.linalg.inv(D)
            # _L = I - torch.matmul(D_, A)
            # L = _L

        return L


    def _make_connection_matrix(self, alpha_dag):
        """構造パラメータから接続行列を作成する

        Args:
            alpha_dag ([type]): 構造パラメータ

        Returns:
            torch.tensor: 接続行列,(n*m行列;nはノード数,mはエッジ数)
        """
        num_edges = 0
        for i in range(len(alpha_dag)):
            if i + 2 < self.window:
                num_edges +=  (i + 2)
            else:
                num_edges += (self.window)
            
                
        B = torch.zeros((self.n_nodes, num_edges))

        e = -1
        for i in range(2, self.n_nodes):
            for m in range(i):
                e = e + 1
                B[i, e] = 1
                B[m, e] = -1

        return B
    
    def _make_neighber_matrix(self, alpha_dag):
        """Make the neighbor matrix from the alpha_dag matrix .

        Args:
            alpha_dag ([type]): [description]

        Returns:
            [type]: [description]
        """
        A = torch.zeros((self.n_nodes, self.n_nodes), dtype=alpha_dag[0].dtype)

        for i in range(2, self.n_nodes):
            alpha = torch.softmax(alpha_dag[i-2], dim=-1)
            for j in range(i):
                # A[i, j] = A[j, i] = alpha_dag[i-2][j, 0]
                A[j, i] = alpha[j, 0]
                # A[j, i] = 1
                # A[i, j] = A[j, i] = 1
        return A
    
    def _make_neighber_matrix2(self, alpha_dag):
        """Make the neighbor matrix from the alpha_dag matrix .

        Args:
            alpha_dag ([type]): [description]

        Returns:
            [type]: [description]
        """
        A = torch.zeros((3*self.n_nodes, 3*self.n_nodes), dtype=alpha_dag[0].dtype)
        offset = 0
        for i in range(0, 3*self.n_nodes):
            if (i % self.n_nodes) in [0, 1]:
                offset += 1
                continue
            alpha = torch.softmax(alpha_dag[i-offset], dim=-1)
            for j in range(i % self.n_nodes):
                A[int((offset/2-1)*self.n_nodes+j), i] = alpha[j, 0]
        return A
    
    def cal_eigen(self, M):
        """行列Mの固有値分解を計算する

        Args:
            M (torch.tensor): 対象の行列
        """
        eigenvalues, eigenvectors = torch.linalg.eig(M)

        return eigenvalues, eigenvectors
    
    def alpha_to_DAG(self, alpha_DAG):
        gene_DAG1 = gt.parse(alpha_DAG[0 * self.n_big_dag: 1 * self.n_big_dag], k=2, window=self.window)
        gene_DAG2 = gt.parse(alpha_DAG[1 * self.n_big_dag: 2 * self.n_big_dag], k=2, window=self.window)
        gene_DAG3 = gt.parse(alpha_DAG[2 * self.n_big_dag: 3 * self.n_big_dag], k=2, window=self.window)

        concat = range(self.n_big_dag, self.n_big_dag + 2)

        return gt.Genotype2(DAG1=gene_DAG1, DAG1_concat=concat,
                            DAG2=gene_DAG2, DAG2_concat=concat,
                            DAG3=gene_DAG3, DAG3_concat=concat)
    
    def cal_cosine_similarity(self, vec1, vec2):
        sim = F.cosine_similarity(vec1, vec2, dim = -1)

        return sim