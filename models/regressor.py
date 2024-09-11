from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation, ConvNormActivation

class ConvTransposed2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.ConvTranspose2d,
        )    

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
            # # 教師のサイズのほうが大きいときは全結合演算でそろえる
            # layers.append(nn.Flatten())
            # operation = nn.Linear(self.guided_channels*self.guided_size*self.guided_size, self.hint_channels*self.hint_size*self.hint_size)
            # layers.append(operation)
            # layers.append(nn.BatchNorm1d(self.hint_channels*self.hint_size*self.hint_size))
            # layers.append(Reshape(hint_channels, hint_size, hint_size))
            
            # 教師のサイズが大きいときは、転置畳み込みでそろえる
            k = self.hint_size - self.guided_size  + 1
            operation = ConvTransposed2dNormActivation(
                self.guided_channels, 
                self.hint_channels, 
                kernel_size=k, 
                stride=1, 
                padding=0, 
                activation_layer=None)
            layers.append(operation)

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
        out = self.regressor_dict["stage"+str(stage)](x)
        return out
    
    def to_device(self, device):
        for i in range(self.num_stages):
            self.regressor_dict["stage"+str(i+1)].to(device)