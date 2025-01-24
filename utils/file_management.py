# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, distribute, and modify this code for non-commercial purposes only.
# Attribution is required. For more details, see:
# https://creativecommons.org/licenses/by-nc/4.0/
#
# Copyright © Shun Miura, 2025

import os
import shutil
import torch


def save_checkpoint(state, ckpt_dir, is_best=False):
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        torch.save(state, best_filename)

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

def load_evaluated_checkpoint_state(model, optimizer, checkpoint_path):
    checkpoint_state = torch.load(checkpoint_path)
    step = checkpoint_state['steps']
    epoch = checkpoint_state['epoch']
    config = checkpoint_state['config']
    if model:
        model.load_state_dict(checkpoint_state['model'], strict=True)
    if optimizer:
        optimizer.load_state_dict(checkpoint_state['optimizer'], strict=True)

    return step, epoch, config