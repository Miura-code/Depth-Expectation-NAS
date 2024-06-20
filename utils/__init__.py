import os
import random
import torch
import numpy as np
import shutil
from utils.preproc import Cutout
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)

def load_checkpoint(model, model_path, device='cuda:0'):
    model = torch.load(model_path, map_location=device)
#    model.load_state_dict(checkpoint.module, strict=True)
    return model

def count_ModelSize_byptflops(model, inputSize):
    # SUMMARY = summary(model, inputSize)

    macs, params = get_model_complexity_info(model, inputSize, as_strings=False,
                                            print_per_layer_stat=True, verbose=False)

    # print(f"TorchInfo summary : \n  {SUMMARY}")
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def set_seed_gpu(seed, gpu):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    cudnn.enabled=True

def get_imagenet(dataset, data_path, cutout_length, validation):
    dataset = dataset.lower()
    dataset == 'imagenet'
    traindir = data_path + '/train'
    validdir = data_path + '/val'

    CLASSES = 1000
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if cutout_length == 0:
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    elif cutout_length > 0:
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(cutout_length),
            ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    trn_data = train_data
    val_data = valid_data
    input_channels = 3
    input_size = 224
    ret = [input_size, input_channels, CLASSES, trn_data]
    if validation:
        ret.append(val_data)

    return ret

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)