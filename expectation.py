from collections import defaultdict

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

# メイン部分
def main():
    # 入力: 係数 alpha を設定
    alpha = {
        (0, 2): 0.1, (1, 2): 0.2,
        (0, 3): 0.3, (1, 3): 0.4, (2, 3): 0.5,
        (1, 4): 0.6, (2, 4): 0.7, (3, 4): 0.8,
        (2, 5): 0.9, (3, 5): 1.0, (4, 5): 1.1,
        (3, 6): 1.2, (4, 6): 1.3, (5, 6): 1.4
    }

    n = 5  # 最大 N の値
    

    # 各 E[X_j] を計算
    for j in range(2, n + 1):
        expectations = {}  # 期待値を格納
        coefficient_counts = defaultdict(int)  # 係数の出現回数

        print("----- {} -----".format(j))
        # calculate_e(j, alpha, expectations, coefficient_counts, 0)
        coefficient_counts = calculate_e_with_recursive_count(j, alpha, coefficient_counts)


        # 結果表示
        print("期待値 E[X_j]:")
        for j, value in expectations.items():
            print(f"E[X_{j}] = {value:.4f}")

        print("\n各係数の出現回数:")
        for key, count in coefficient_counts.items():
            print(f"alpha_{key} 出現回数: {count}")

        # print("係数の合計: {}".format(sum(coefficient_counts.items())))
        print(sum(coefficient_counts.values()))

if __name__ == "__main__":
    main()
