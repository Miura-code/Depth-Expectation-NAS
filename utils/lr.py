import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, eta_min=0, last_epoch=-1):
        self.eta_min = eta_min
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        # if self.last_epoch < 10:
        #     return [lr * (1 - 0.1 * self.last_epoch) for lr in self.base_lrs]
        # else:
        #     return [lr * 0.1 for lr in self.base_lrs]

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch > 0 and self.last_epoch <= 50:
            # 0～50エポック: 学習率低下（コサイン曲線）
            lr = [self.eta_min + (base_lr - self.eta_min) * 
                  (1 + math.cos(math.pi * self.last_epoch / 50)) / 2 
                  for base_lr, group in
                  zip(self.base_lrs, self.optimizer.param_groups)]
        elif self.last_epoch > 50 and self.last_epoch <= 60:
            # 50～60エポック: 学習率増加（コサイン曲線）
            lr = [self.eta_min + (base_lr - self.eta_min) * 
                  (1 - math.cos(math.pi * (self.last_epoch - 50) / 10)) / 2 
                  for base_lr, group in
                  zip(self.base_lrs, self.optimizer.param_groups)]
        elif self.last_epoch > 60 and self.last_epoch <= 150:
            # 60～150エポック: 学習率低下（コサイン曲線）
            lr = [self.eta_min + (base_lr - self.eta_min) * 
                  (1 + math.cos(math.pi * (self.last_epoch - 60) / 90)) / 2 
                  for base_lr, group in
                  zip(self.base_lrs, self.optimizer.param_groups)]
        else:
            # 150エポック以降: 学習率固定（最小値）
            lr = [self.eta_min
                  for base_lr, group in
                  zip(self.base_lrs, self.optimizer.param_groups)]  # 最小値の学習率
        return lr