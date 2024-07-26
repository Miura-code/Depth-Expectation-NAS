# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import os
import time
import numpy as np
import torch
from torch.cuda import current_blas_handle, device
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.data_util import get_data
from utils.eval_util import AverageMeter, accuracy

from utils.data_prefetcher import data_prefetcher

from models.search_stage import SearchShareStageController
from models.architect import Architect

from trainer.searchStage_trainer import SearchStageTrainer_WithSimpleKD

class SearchShareStageTrainer(SearchStageTrainer_WithSimpleKD):
    def __init__(self, config) -> None:
        super().__init__(config)

    def construct_model(self):
        """get data loader"""
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=0, validation=False
        )

        n_train = len(train_data)
        split_half = n_train // 2
        split = int(np.floor(self.config.train_portion * n_train))
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split_half:split_half+split])
        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=self.config.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.config.workers,
                                                        pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=self.config.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.config.workers,
                                                        pin_memory=True)
        
        """build model"""
        print("init model")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        model = SearchShareStageController(input_channels, self.config.init_channels, n_classes, self.config.layers, self.criterion, self.config.genotype, device_ids=self.config.gpus, spec_cell=self.config.spec_cell)
        self.model = model.to(self.device)
        print("init model end!")

        """build optimizer"""
        print("get optimizer")
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), self.config.alpha_lr, betas=(0.5, 0.999), weight_decay=self.config.alpha_weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        self.architect = Architect(self.model, self.config.w_momentum, self.config.w_weight_decay)

    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_lr = self.lr_scheduler.get_last_lr()[0]

        self.model.print_alphas(self.logger)
        self.model.train()
        prefetcher_trn = data_prefetcher(self.train_loader)
        prefetcher_val = data_prefetcher(self.valid_loader)
        trn_X, trn_y = prefetcher_trn.next()
        val_X, val_y = prefetcher_val.next()
        i = 0
        while trn_X is not None:
            i += 1
            N = trn_X.size(0)
            self.steps += 1

            # architect step (alpha)
            self.alpha_optim.zero_grad()
            self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim)
            self.alpha_optim.step()

            self.alpha_optim.zero_grad()
            alpha = self.architect.net.alpha_DAG
            self.n_nodes = self.config.layers // 3
            d_depth = self.cal_depth(alpha, self.n_nodes, 3)
            depth_loss = -1.0 * d_depth
            depth_loss.backward()
            self.alpha_optim.step()
            
            # child network step (w)
            self.w_optim.zero_grad()
            logits = self.model(trn_X)
            loss = self.model.criterion(logits, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if self.steps % self.log_step == 0:
                self.writer.add_scalar('train/lr', round(cur_lr, 5), self.steps)
                self.writer.add_scalar('train/loss', loss.item(), self.steps)
                self.writer.add_scalar('train/top1', prec1.item(), self.steps)
                self.writer.add_scalar('train/top5', prec5.item(), self.steps)

            if i % self.config.print_freq == 0 or i == len(self.train_loader) - 1:
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                        )
            
            trn_X, trn_y = prefetcher_trn.next()
            val_X, val_y = prefetcher_val.next()
        
        printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, losses.avg
