from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 再帰的に期待値を計算する関数
def calculate_e(j, alpha, expectations, coefficient_counts, n_loop):
    print("called {} -- :".format(j))
    """
    再帰的に期待値 E[X_j] を計算し、係数の出現回数を正確に追跡。

    Parameters:
        j: int
            現在のインデックス j。
        alpha: dict
            係数 alpha_{i,j} を格納する辞書。
        expectations: dict
            計算済みの期待値を格納する辞書。
        coefficient_counts: dict
            各 alpha_{i,j} の出現回数を格納する辞書。
    
    Returns:
        float
            E[X_j] の計算結果。
    """
    # 基底条件: E[X_0] = E[X_1] = 0
    if j <= 1:
        return 0, n_loop
    
    # # 再帰計算済みの場合でも、カウント処理を行う必要がある
    # if j in expectations:
    #     loop = 0
    #     for k in range(max(0, j - 3), j):  # j-3, j-2, j-1 の係数をカウント
    #         if (k, j) in alpha:
    #             coefficient_counts[(k, j)] += 1
    #             loop += 1   
    #             print("loop : {}".format(loop))
    #             print("coefficient_counts[({}, {})] : {}".format(k,j,coefficient_counts[(k, j)]))
    #     return expectations[j], loop

    # 再帰的に計算
    
    loop = 0
    value = 0
    for k in range(max(0, j - 3), j):  # k = j-3, j-2, j-1
        if (k, j) in alpha:
            e, n_loop = calculate_e(k, alpha, expectations, coefficient_counts, loop)
            loop = n_loop + 1
            print("loop : {}".format(loop))
            value += alpha[(k, j)] * (e + 1)
            coefficient_counts[(k, j)] += n_loop+1  # 使用回数を増加
            print("coefficient_counts[({}, {})] : {}".format(k,j,coefficient_counts[(k, j)]))

    # 計算結果を保存
    expectations[j] = value
    return value, loop

def calculate_e_with_recursive_count(j, alpha, coefficient_counts):
    print("called {} -- :".format(j))

    """
    再帰的に期待値 E[X_j] を計算し、係数の出現回数を追跡。
    再帰的な寄与もすべて考慮する。

    Parameters:
        j: int
            現在のインデックス j。
        alpha: dict
            係数 alpha_{i,j} を格納する辞書。
        coefficient_counts: dict
            各 alpha_{i,j} の出現回数を格納する辞書。
    
    Returns:
        float
            E[X_j] の計算結果。
    """
    # 基底条件: E[X_0] = E[X_1] = 0
    if j <= 1:
        return coefficient_counts
    # 再帰的な計算
    for k in range(max(0, j - 3), j):  # k = j-3, j-2, j-1
        if (k, j) in alpha:
            # 再帰呼び出しを行い、再帰的なカウントを伝播
            sub_coefficient_counts = defaultdict(int)
            sub_coefficient_counts = calculate_e_with_recursive_count(k, alpha, sub_coefficient_counts)

            # 係数のカウントに再帰的な寄与を追加
            loop = 0
            for key, count in sub_coefficient_counts.items():
                loop += 1
                coefficient_counts[key] += count
                print("coefficient_counts[{}] : {}".format(key,coefficient_counts[key]))


            # 現在の係数をカウント（直接的な寄与を追加）
            coefficient_counts[(k, j)] += loop + 1
            print("coefficient_counts[({},{})] : {}".format(k,j,coefficient_counts[(k,j)]))

    return coefficient_counts

def loss(alpha, beta):
    depth_list = torch.zeros((1, n_big_nodes+2))
    alpha = [alpha for alpha in alpha]
    beta = [F.softmax(beta, dim=0) for beta in beta]

    alpha_dag = alpha[0 * n_big_nodes: (0+1) * n_big_nodes]
    depth_list[0] = _expectation_dp(alpha_dag, depth_list[0])
    depth = 0
    offset = 0
    for i in range(2, n_big_nodes + 1):
        for j in range(i+1, n_big_nodes + 2):
            depth += beta[0][offset] * (depth_list[0][i] + depth_list[0][j])
            offset += 1

    return depth.item(), depth_list

def _expectation_dp(alpha, ExpectedDepth):
    for j in range(n_big_nodes+2):
        if j == 0 or j == 1:
            ExpectedDepth[j] = 0
        elif j < window:
            edge_max, _ = torch.topk(alpha[j-2][:,:-1], 1)
            # print("alpha[{}] : {}".format(j-2, alpha[j-2]))
            # print("edge_max : ", edge_max)
            edge_max = F.softmax(edge_max, dim=0)
            # print("edge_max : ", edge_max)
            # input()
            for i in range(j):
                ExpectedDepth[j] += edge_max[i][0] * ExpectedDepth[i]
            ExpectedDepth[j] += 1
        else:
            edge_max, _ = torch.topk(alpha[j-2][:,:-1], 1)
            # print("alpha[{}] : {}".format(j-2, alpha[j-2]))
            # print("edge_max : ", edge_max)
            edge_max = F.softmax(edge_max, dim=0)
            # print("edge_max : ", edge_max)
            # input()
            for s, i in enumerate(range(j-window, j)):
                ExpectedDepth[j] += edge_max[s][0] * (ExpectedDepth[i])
            ExpectedDepth[j] += 1

    return ExpectedDepth

# メイン部分
def main():
    beta = nn.ParameterList()
    for _ in range(1):
        beta.append(nn.Parameter(1.0 * torch.ones(int((n_big_nodes)*(n_big_nodes-1)/2))))

    alpha = nn.ParameterList()
    for _ in range(1):
        for i in range(n_big_nodes):
            # sliding window
            if i + 2 < window:
                alpha.append(nn.Parameter(1.0 * torch.ones(i + 2, n_ops)))
            else:
                alpha.append(nn.Parameter(1.0 * torch.ones(window, n_ops)))

    # alphaの特定の値を大きくしてlossの変化を調べる
    depth_values = []

    scaling_factors = torch.linspace(1, 1.1, steps=10)  # スケールの範囲
    # プロット
    plt.figure(figsize=(10, 6))
    
    
    # m = 0
    # # for l in range(int((n_big_nodes)*(n_big_nodes-1)/2)):
    # for l in range(1,2):
    # # for l in [1,2,3,4,5]:
    #     depth_values.append([])
        
    #     for scale in scaling_factors:
    #         print("scale = {}".format(scale))
    #         for m in [0, 1]:
    #             base_alpha = [param.detach().clone() for param in alpha]
    #             base_beta = [param.detach().clone() for param in beta]
    #             print("m = {}".format(m))
    #             # alphaの特定の値をスケールアップ
    #             base_alpha[l][m][0].data *= scale
    #             print("base_alpha[{}].data : {}".format(l, base_alpha[l].data))

    #             # base_beta[0][l] *= scale
    #             # print(base_beta[0].data)

    #             # lossを計算
    #             depth_expectation, _ = loss(base_alpha, base_beta)
    #             depth_values[-1].append(depth_expectation)

    #             print(depth_values)
    #         # plt.plot(scaling_factors, depth_values[-1], marker='o', label="Alpha[{}][{}]".format(l,m))

    m=2
    # for l in range(1,n_big_nodes):
    for l in range(int((n_big_nodes)*(n_big_nodes-1)/2)):
    # for l in [1,2,3,4,5]:
        depth_values.append([])
        base_alpha = [param.detach().clone() for param in alpha]
        base_beta = [param.detach().clone() for param in beta]
        for scale in scaling_factors:
            print("scale = {}".format(scale))
           
            # alphaの特定の値をスケールアップ
            # base_alpha[l][m][0].data *= scale
            # print("base_alpha[{}].data : {}".format(l, base_alpha[l].data))

            base_beta[0][l] *= scale
            print(base_beta[0].data)

            # lossを計算
            depth_expectation, _ = loss(base_alpha, base_beta)
            depth_values[-1].append(depth_expectation)

        # plt.plot(scaling_factors, depth_values[-1], marker='o', label="Beta[{}][{}]".format(l,m))
        plt.plot(scaling_factors, depth_values[-1], marker='o', label="Beta[{}]".format(l))

    # plt.xlabel("Alpha Scaling Factor")
    # plt.ylabel("Depth Expectation")
    # plt.title("Effect of Scaling Alpha on Depth Expectation")
    # plt.legend()
    # plt.grid()
    # plt.savefig("./assets/Expectation_alpha_m={}.png".format(m))
    
    plt.xlabel("Beta Scaling Factor")
    plt.ylabel("Depth Expectation")
    plt.title("Effect of Scaling Beta on Depth Expectation")
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10, ncol=3)
    plt.grid()
    plt.savefig("./assets/Expectation_beta_.png", bbox_inches='tight')

    

if __name__ == "__main__":
    n_big_nodes = 10
    window=3
    n_ops=4
    main()
