import torch
import torch.nn as nn
import torch.nn.functional as F

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
        soft_loss = self.soft_criteria(s_logits, t_logits)
        loss = self.l * hard_loss + (1 - self.l) * self.T * self.T * soft_loss

        if return_detail:
            return hard_loss, soft_loss, loss
                
        return loss

class HintLoss(nn.Module):
    '''
    FitNets: Hint for Thin Deep Nets
    '''
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        '''
        logits: 生徒モデルの中間層出力
        targets: 教師モデルの中間層出力
        '''
        loss = nn.MSELoss()
        kd_loss = loss(logits, targets)
        return kd_loss