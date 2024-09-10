from collections import OrderedDict
from typing import List
import torch
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape  # 元の形状を保存

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class Regressor(nn.Module):
    """
    DAG for search
    Each edge is mixed and continuous relaxed
    """
    def __init__(self, guided_size, hint_size, guided_channels, hint_channels):
        """
        guided_size: guided(student) features size
        hint_size: hint(teacher) features size
        channels: # of hint feature channels
        """
        super().__init__()
        self.guided_size = guided_size
        self.hint_size = hint_size
        self.guided_channels = guided_channels
        self.hint_channels = hint_channels
        
        layers: List[nn.Module] = []
        if self.hint_size < self.guided_size or (self.hint_channels > self.guided_channels and self.hint_size == self.guided_size) :
            # Conv層でサイズ、チャンネルをそろえる
            k = self.guided_size - self.hint_size + 1
            operation = Conv2dNormActivation(
                self.guided_channels, 
                self.hint_channels, 
                kernel_size=k, 
                stride=1, 
                padding=0, 
                activation_layer=None)
            layers.append(operation)
        elif self.hint_size == self.guided_size and self.hint_channels == self.guided_channels:
            # サイズ、チャンネルが同じときは何もしない
            layers.append(nn.Identity())
        elif (self.hint_size > self.guided_size):
            # 教師のサイズのほうが大きいときは全結合演算でそろえる
            layers.append(nn.Flatten())
            operation = nn.Linear(self.guided_channels*self.guided_size*self.guided_size, self.hint_channels*self.hint_size*self.hint_size)
            layers.append(operation)
            layers.append(nn.BatchNorm1d(self.hint_channels*self.hint_size*self.hint_size))
            layers.append(Reshape(hint_channels, hint_size, hint_size))

        self.features = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.features(x)


class Regressor_Controller(nn.Module):
    def __init__(self, num_stages:int=1, reg_info_set:list[tuple]=[(14, 28, 128, 64)]):
        """
        Args:
            num_stages: Regressorの数、Hint学習を行う中間層の数とおなじ
            reg_info_set: Regressorにわたす引数のセット、Regressorの数と一緒である必要がある
                [("guided_size", "hint_size", "guided_channels", "hint_channels")]
        """
        super(Regressor_Controller, self).__init__()
        
        if num_stages != len(reg_info_set):
            raise ValueError("Regressorの数と引数セットの数が一致しません.\n"
                             f"{{{num_stages}}} Regressors, and info = {{{reg_info_set}}}")
        self.num_stages = num_stages
        
        self.regressor_dict = {}
        for i in range(self.num_stages):
            set = reg_info_set[i]
            regressor = Regressor(set[0], set[1], set[2], set[3])
            self.regressor_dict["stage"+str(i+1)] = regressor
            
        self.curr_stage = 0
            
    def forward(self, x, stage):
        return self.regressor_dict["stage"+str(stage)](x)
    
    def to_device(self, device):
        for i in range(self.num_stages):
            self.regressor_dict["stage"+str(i+1)].to(device)