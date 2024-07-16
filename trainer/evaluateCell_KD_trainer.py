# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import teacher_models
from timm_.loss.distillation_losses import KD_Loss, SoftTargetKLLoss
import utils
from utils.data_util import get_data, split_dataloader
from utils.file_management import load_teacher_checkpoint_state
from utils.params_util import collect_params
from utils.eval_util import AverageMeter, accuracy
from models.augment_cellcnn import AugmentCellCNN
from utils.data_prefetcher import data_prefetcher
from utils.visualize import showModelOnTensorboard
class EvaluateCellTrainer_WithSimpleKD():
    def __init__(self, config):
        self.config = config

        self.gpu = self.config.gpus
        self.device = utils.set_seed_gpu(config.seed, config.gpus)

        """get the train parameters"""
        self.total_epochs = self.config.epochs
        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.max_lr = self.config.lr

        self.T = self.config.T
        self.l = self.config.l

        """construct the whole network"""
        self.resume_path = self.config.resume_path

        """save checkpoint path"""
        self.save_epoch = 1
        self.ckpt_path = self.config.path

        """log tools in the running phase"""
        self.steps = 0
        self.log_step = 10
        self.logger = self.config.logger
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.path, "tb"))
        self.writer.add_text('config', config.as_markdown(), 0)

        self.construct_model()


    def construct_model(self):
        # ================= define data loader ==================
        input_size, input_channels, n_classes, train_data, valid_data = get_data(
            self.config.dataset, self.config.data_path, self.config.cutout_length, validation=True
        )
        self.train_loader, self.valid_loader = split_dataloader(train_data, self.config.train_portion, self.config.batch_size, self.config.workers)
        
        print("init model")
        # ================= define criteria ==================
        self.hard_criterion = nn.CrossEntropyLoss().to(self.device)
        self.soft_criterion = SoftTargetKLLoss(self.T).to(self.device)
        self.criterion = KD_Loss(self.soft_criterion, self.hard_criterion, self.l, self.config.T)
        self.use_aux = self.config.aux_weight > 0.
        # ================= Student model ==================
        model = AugmentCellCNN(input_size, input_channels, self.config.init_channels, n_classes, self.config.layers, self.use_aux, self.config.genotype)
        self.model = model.to(self.device)
        # ================= Teacher Model ==================
        teacher_model = self.load_teacher(n_classes)
        self.teacher_model = teacher_model.to(self.device)

        # showModelOnTensorboard(self.writer, self.model, self.train_loader)
        showModelOnTensorboard(self.writer, self.teacher_model, self.train_loader)
        print("init model end!")

        # ================= build Optimizer ==================
        print("get optimizer")
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_epochs, eta_min=self.config.lr_min)
    
    def load_teacher(self, n_classes):
        """
            load pretrained teacher model
            and Freeze all parameter to be not learnable
        """
        try:
            model = teacher_models.__dict__[self.config.teacher_name](num_classes = n_classes)
        except (RuntimeError, KeyError) as e:
            self.logger.info("model loading error!: {}\n \
                        tring to load from torchvision.models".format(e))
        model = torchvision.models.__dict__[self.config.teacher_name](num_classes = n_classes)

        _, _ = load_teacher_checkpoint_state(model=model, optimizer=None, checkpoint_path=self.config.teacher_path)
        
        for name, param in model.named_parameters():
            param.requires_grad = False
        self.logger.info(f"--> Loaded teacher model '{self.config.teacher_name}' from '{self.config.teacher_path}' and Freezed parameters)")
        
        return model
    
    def resume_model(self, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logger.info("--> No loaded checkpoint!")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(f"--> Loaded checkpoint '{model_path}' (epoch {self.start_epoch})")
    
    def save_checkpoint(self, epoch, is_best=False):
        if epoch % self.save_epoch == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                    }
            # filename = os.path.join(self.ckpt_path, f'{epoch}_ckpt.pth.tar')
            # torch.save(state, filename)
            if is_best:
                best_filename = os.path.join(self.ckpt_path, 'best.pth.tar')
                torch.save(state, best_filename)
    
    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        hard_losses = AverageMeter()
        soft_losses = AverageMeter()
        cur_lr = self.optimizer.param_groups[0]['lr']
        
        self.model.train()
        self.teacher_model.train()

        prefetcher = data_prefetcher(self.train_loader)
        X, y = prefetcher.next()
        i = 0
        while X is not None:
            i += 1
            N = X.size(0)
            self.steps += 1

            # ================= optimize network parameter ==================
            teacher_guide = self.teacher_model(X)
            logits, aux_logits = self.model(X)
            hard_loss, soft_loss, loss = self.criterion(logits, teacher_guide, y, True)
            if self.use_aux:
                loss += self.config.aux_weight * self.hard_criterion(aux_logits, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            prec1, prec5 = accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            hard_losses.update(hard_loss.item(), N)
            soft_losses.update(soft_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if self.gpu == 0 and (i % self.config.print_freq == 0 or i == len(self.train_loader) - 1):
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Hard Loss {hard_losses.val:.4f} ({hard_losses.avg:.4f})\t'
                        f'Soft Loss {soft_losses.val:.4f} ({soft_losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                    )

            X, y = prefetcher.next()

        if self.gpu == 0:
            printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))

        return top1.avg, hard_losses.avg, soft_losses.avg, losses.avg
    
    def val_epoch(self, epoch, printer):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        self.model.eval()

        prefetcher = data_prefetcher(self.valid_loader)
        X, y = prefetcher.next()
        i = 0

        with torch.no_grad():
            while X is not None:
                N = X.size(0)
                i += 1

                logits, _ = self.model(X)
                loss = self.hard_criterion(logits, y)

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
                
                if (i % self.config.print_freq == 0 or i == len(self.valid_loader) - 1):
                    printer(f'Valid: Epoch: [{epoch}][{i}/{len(self.valid_loader)}]\t'
                            f'Step {self.steps}\t'
                            f'Loss {losses.avg:.4f}\t'
                            f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})')
                
                X, y = prefetcher.next()
                
            printer("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, losses.avg

