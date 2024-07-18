from teacher_models.utils.alert import *
from teacher_models.utils.layer_replace import *
from teacher_models.utils.optimizer_params import *

import torch
import torch.nn as nn

def freeze_model(model, unfreeze: bool = False):
    """
        freeze the model parameter excepting last classification layer
        Args:
            model: ニューラルネットモデル
            unfreeze: 指定することでモデルのすべての層の学習を開始する
    """
    if unfreeze:
        for name, params in model.named_parameters():
            params.require_grad = True
    else:
        classifier_layer = model.get_classifier()
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in classifier_layer.named_parameters():
            param.requires_grad = True