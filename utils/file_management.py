
import os
import shutil
import torch


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

def load_teacher_checkpoint_state(model, optimizer, checkpoint_path):
    checkpoint_state = torch.load(checkpoint_path)
    if model:
        model.load_state_dict(checkpoint_state['model'], strict=True)
    if optimizer:
        optimizer.load_state_dict(checkpoint_state['w_optim'], strict=True)
    step = checkpoint_state['steps']
    epoch = checkpoint_state['epoch']

    return step, epoch