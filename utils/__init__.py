import os
import random
import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import shutil
import torch.backends.cudnn as cudnn

from utils import setting
from .data_prefetcher import *
from .data_util import *
from .eval_util import *
from .file_management import *
from .graphs  import *
from .imagenet_loader import *
from .logging_util import *
from .loss import *
from .measurement_utils import *
from .params_util import *
from .parser import *
from .preproc import *
from .visualize import *

class SETTING():
    def __init__(self):
        for attr in dir(setting):
            if attr.isupper():
                setattr(self, attr, getattr(setting, attr))

def set_seed_gpu(seed, gpus:list):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    cudnn.benchmark = True
    cudnn.enabled=True
    if torch.cuda.is_available():
      torch.cuda.set_device(gpus[0])
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    return device
    

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.isdir((path)):
        os.makedirs((path))
        print("make dirs")

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def ListToMarkdownTable(name_list, value_list):
  """ Return list as markdown format of table"""
  text = "|name|value|  \n|-|-|  \n"
  for attr, value in zip(name_list, value_list):
      text += "|{}|{}|  \n".format(attr, value)

  return text

def grad_check(model, layer_name):
  for name, param in model.named_parameters():
    if layer_name not in name:
      continue
    if param.grad is not None:
        print(f"{name}の勾配:\n{param.grad}")
    else:
        print(f"{name}の勾配はありません")