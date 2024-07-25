import os
import random
import torch
import numpy as np
import shutil
from utils.preproc import Cutout
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

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