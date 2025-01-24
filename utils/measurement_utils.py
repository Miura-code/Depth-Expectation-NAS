# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, distribute, and modify this code for non-commercial purposes only.
# Attribution is required. For more details, see:
# https://creativecommons.org/licenses/by-nc/4.0/
#
# Copyright © Shun Miura, 2025

import json
import numpy as np
import time

from ptflops import get_model_complexity_info
from thop import profile
from thop import clever_format
from icecream import ic

class TimeKeeper():
    def __init__(self):
        self.start_time = time.time()
    def end(self):
        self.end_time = time.time()
        self.time_diff = self.end_time - self.start_time
    def print_info(self):
        return self.start_time, self.end_time, self.time_diff

def count_ModelSize_byptflops(model, inputSize, path="./flops_info.txt"):
    # SUMMARY = summary(model, inputSize)
    
    with open(path, mode='w') as f:
        macs, params = get_model_complexity_info(model, inputSize, as_strings=True,
                                                print_per_layer_stat=True, verbose=False,
                                                ost=f)

    # print(f"TorchInfo summary : \n  {SUMMARY}")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_ModelSize_bythop(model, inputSize, path="./thop_info.json"):
    # SUMMARY = summary(model, inputSize)
    
    macs, params, ret_dict = profile(model, inputs=(inputSize,), ret_layer_info=True)
    # macs, params = clever_format([macs, params], "%.3f")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ret_dict, f, ensure_ascii=False, indent=2, separators=(',', ':'))

    return macs, params, ret_dict

def count_ModelSize_exclude_extra(n_big_node, mac, params, ret_dict, ignore_layer, dead_cells):
    ex_mac = sum([ret_dict[layer][0] for layer in ignore_layer])
    ex_params = sum([ret_dict[layer][1] for layer in ignore_layer])

    for s, dead_cell in enumerate(dead_cells):
        ex_mac += ret_dict["bigDAG{}".format(s+1)][2]["cells"][0]
        ex_params += ret_dict["bigDAG{}".format(s+1)][2]["cells"][1]
        for cell in dead_cell:
            ex_mac += ret_dict["bigDAG{}".format(s+1)][2]["bigDAG"][2]["{}".format(cell)][0]
            ex_mac += ret_dict["bigDAG{}".format(s+1)][2]["bigDAG"][2]["{}".format(n_big_node+cell)][0]
            ex_params += ret_dict["bigDAG{}".format(s+1)][2]["bigDAG"][2]["{}".format(cell)][1]
            ex_params += ret_dict["bigDAG{}".format(s+1)][2]["bigDAG"][2]["{}".format(n_big_node+cell)][1]

    mac -= ex_mac
    params -= ex_params

    return mac, params

def MACs_float_to_ratio(values:list):
    total = sum(values)
    ratios = [i/total for i in values]
    
    return ratios