import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCombinedLoss(nn.Module):
    def __init__(self, loss_function_weight_pairs):
        super(WeightedCombinedLoss, self).__init__()
        # 損失関数とその重みのペアを保持
        self.loss_function_weight_pairs = loss_function_weight_pairs

    def forward(self, *inputs_list, detail=True):
        """
        各損失関数に対応する入力をリストで受け取り、対応する損失関数に渡す
        :param inputs_list: 損失関数ごとに渡す入力をまとめたリスト
        """
        total_loss = 0
        losses = []
        # 損失関数、重み、対応する入力を処理
        for (loss_fn, weight), (inputs) in zip(self.loss_function_weight_pairs, inputs_list):
            # 各損失関数に動的に対応する入力を渡す
            loss = loss_fn(*inputs)
            total_loss += weight * loss
            losses.append(loss)

        if detail:
            losses.append(total_loss)
            return losses
        
        return total_loss

class SoftTargetKLLoss(nn.Module):
    r"""
    KL divergence between two distribution which is represented by soft target of labels
    args:
        T:templature
    """
    def __init__(self, T):
        super(SoftTargetKLLoss, self).__init__()
        self.T = T

    def forward(self, logits, targets):
        '''
        logits: 予測結果(ネットワークの出力)
        targets: 
        F.kl_div()は,inputはlogで渡す必要が有り、targetは内部でlogに変換される
        '''
        logits = logits / self.T
        targets = targets / self.T

        loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(targets, dim=-1))
        return loss
    
class KD_Loss(nn.Module):
    r"""
        Simple KD loss introduced by 
        "Hinton, G.E., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. ArXiv, abs/1503.02531."
        func forward() <- (student_logits, teacher_logits, targets)
    """
    def __init__(self, soft_criteria, hard_criteria, l, T):
        super(KD_Loss, self).__init__()
        self.soft_criteria = soft_criteria
        self.hard_criteria = hard_criteria
        self.l = l
        self.T = T

    def forward(self, s_logits, t_logits, targets, return_detail=False):
        hard_loss = self.hard_criteria(s_logits, targets)
        soft_loss = self.T * self.T *self.soft_criteria(s_logits, t_logits)
        loss = self.l * hard_loss + (1 - self.l) * soft_loss

        if return_detail:
            return hard_loss, soft_loss, loss
                
        return loss

class HintLoss(nn.Module):
    '''
    FitNets: Hint for Thin Deep Nets
    '''
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, logits, targets):
        '''
        logits: 生徒モデルの中間層出力
        targets: 教師モデルの中間層出力
        '''
        
        kd_loss = self.loss(logits, targets)
        return kd_loss
    
class AlphaArchLoss(nn.Module):
    """二つの構造パラメータの損失を計算する
    """
    def __init__(self, target_alphaDAG):
        super().__init__()
        self.target_alphaDAG = target_alphaDAG
        
    def forward(self, alphaDAG):
        loss = 0
        for alpha, target_alpha in zip(alphaDAG, self.target_alphaDAG):
            loss += torch.sum((target_alpha - alpha) ** 2)
        return loss
    
class AlphaArchLoss_Temprature(nn.Module):
    """二つの構造パラメータの損失を計算する
    """
    def __init__(self, target_alphaDAG, T):
        super().__init__()
        self.target_alphaDAG = target_alphaDAG
        self.T = T

        self.soft_target_alpha = []
        for target_alpha in self.target_alphaDAG:
            self.soft_target_alpha.append(torch.softmax(target_alpha / self.T, dim=-1))
        
    def forward(self, alphaDAG):
        loss = torch.tensor(0.0, requires_grad=True, device=alphaDAG[0].device)
        for alpha, target_alpha in zip(alphaDAG, self.soft_target_alpha):
            loss = loss + torch.sum((target_alpha - alpha) ** 2)
        return (self.T **2)  * loss
        
    
class AlphaLaplacianLoss(nn.Module):

    # TODO: backwardが計算できないなんで、
    def __init__(self, window, n_nodes, target_alphaDAG=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.window = window
        
        self.tvalue = torch.empty([])
        if target_alphaDAG is not None:
            tL1 = self._Laplacian_from_genotype(target_alphaDAG[:6])
            tL2 = self._Laplacian_from_genotype(target_alphaDAG[6:12])
            tL3 = self._Laplacian_from_genotype(target_alphaDAG[12:])
            tvalue1, _ = self._cal_eigen(tL1)
            tvalue2, _ = self._cal_eigen(tL2)
            tvalue3, _ = self._cal_eigen(tL3)
        
            self.tvalue = torch.concat([tvalue1, tvalue2, tvalue3])
    
    def re__init__(self, target_alphaDAG):
        self.__init__(self.window, self.n_nodes, target_alphaDAG)

    def forward(self, alphaDAG, target_alphaDAG=None):
        L1 = self._Laplacian_from_genotype(alphaDAG[:6])
        L2 = self._Laplacian_from_genotype(alphaDAG[6:12])
        L3 = self._Laplacian_from_genotype(alphaDAG[12:])
        value1, _ = self._cal_eigen(L1)
        value2, _ = self._cal_eigen(L2)
        value3, _ = self._cal_eigen(L3)
        
        value = torch.concat([value1, value2, value3])

        if target_alphaDAG is not None:
            tL1 = self._Laplacian_from_genotype(target_alphaDAG[:6])
            tL2 = self._Laplacian_from_genotype(target_alphaDAG[6:12])
            tL3 = self._Laplacian_from_genotype(target_alphaDAG[12:])
            tvalue1, _ = self._cal_eigen(tL1)
            tvalue2, _ = self._cal_eigen(tL2)
            tvalue3, _ = self._cal_eigen(tL3)
            
            self.tvalue = torch.concat([tvalue1, tvalue2, tvalue3])
            
        similarity = self._cal_cosine_similarity(torch.abs(value), torch.abs(self.tvalue))
        
        return -similarity
    
    def _Laplacian_from_genotype(self, alpha_dag, C=False):
        """構造パラメータからグラフラプラシアン行列を作成する

        Args:
            alpha_dag ([type]): 構造パラメータ（１ステージ分）

        Returns:
            torch.tensor: グラフラプラシアン
        """
        if C:
            B = self._make_connection_matrix(alpha_dag)
            W = []
            for alphas_node in alpha_dag:
                alphas_node = torch.softmax(alphas_node, dim=-1)
                for alpha in alphas_node:
                    W.append(alpha[:1])
                    # W.append(torch.tensor([1.]))
            W = torch.diag(torch.concat(W))

            L = torch.matmul(torch.matmul(B, W), torch.t(B))
        else:
            A = self._make_neighber_matrix(alpha_dag)

            D = torch.diag(A.sum(dim=1))
            L = D - A

        return L

    def _cal_eigen(self, M):
        """行列Mの固有値分解を計算する

        Args:
            M (torch.tensor): 対象の行列
        """
        eigenvalues, eigenvectors = torch.linalg.eig(M)

        return eigenvalues, eigenvectors
    
    def _cal_cosine_similarity(self, vec1, vec2):
        sim = F.cosine_similarity(vec1, vec2, dim = -1)

        return sim
    
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
            
                
        B = torch.zeros((self.n_nodes, num_edges), requires_grad=True)

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
    
class CosineScheduler():
    """コサイン曲線に従って値を減衰あるいは増加させる
    """
    def __init__(self, initial_value, final_value, total_steps):
        """初期化

        Args:
            initial_value (_type_):初期値
            final_value (_type_): 最終的な値
            total_steps (_type_): 学習エポック数
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
    
    def get_decay_value(self, current_step):
        """コサイン曲線に従って値を減衰させる

        Args:
            current_step (_type_): 現在のステップ数
        """
        if current_step >= self.total_steps:
            return self.final_value
        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / self.total_steps))
        decayed_value = self.final_value + (self.initial_value - self.final_value) * cosine_decay
        return decayed_value
    
    def get_increase_value(self, current_step):
        """コサイン曲線に従って値を増加させる

        Args:
            current_step (_type_): 現在のステップ数
        """
        if current_step >= self.total_steps:
            return self.final_value
        cosine_increase = 0.5 * (1 - math.cos(math.pi * current_step / self.total_steps))
        increased_value = self.initial_value + (self.final_value - self.initial_value) * cosine_increase
        return increased_value