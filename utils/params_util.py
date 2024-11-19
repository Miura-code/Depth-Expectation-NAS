# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch

from genotypes.genotypes import parse_dag_to_alpha


def collect_params(model_list, exclude_bias_and_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    param_list = []
    for model in model_list:
        for name, param in model.named_parameters():
            if exclude_bias_and_bn and ('bn' in name or 'bias' in name):
                param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
            else:
                param_dict = {'params': param}
            param_list.append(param_dict)
    return param_list

def resume_model(model, model_path=None, device='cpu', printer=print):
    if not model_path:
        start_epoch = 0
        printer("--> No loaded checkpoint!")
    else:
        checkpoint = torch.load(model_path, map_location=device)

        model.load_state_dict(checkpoint['model'], strict=True)
        printer(f"--> Loaded checkpoint '{model_path}'")
    return model

def resume_alpha_discrete(model, DAG, model_path=None, device='cpu', sw=10, printer=print):
    current_state_dict = model.state_dict()

    discrete_alpha_list = parse_dag_to_alpha(DAG, n_ops=model.alpha_DAG[0][0].size(0), window=sw, device=device)

    count = 0
    for name, param in model.named_parameters():
        if 'alpha' in name:
            model.alpha_DAG[count] = current_state_dict[name] = discrete_alpha_list[count]
            count += 1

    model.load_state_dict(current_state_dict, strict=True)
    printer(f"--> Loaded DISCRETED checkpoint '{model_path}'")
    
    return model

def freeze_alphaParams(model, printer=print):
    for name, param in model.named_parameters():
        if 'alpha' in name or 'beta' in name:
            param.requires_grad = False
            print(param)
    printer(f"--> Loaded alpha parameters are Freezed")

    return model