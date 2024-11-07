import numpy as np
import time

from ptflops import get_model_complexity_info

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
                                                print_per_layer_stat=False, verbose=False,
                                                ost=f)

    # print(f"TorchInfo summary : \n  {SUMMARY}")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def MACs_float_to_ratio(values:list):
    total = sum(values)
    ratios = [i/total for i in values]
    
    return ratios