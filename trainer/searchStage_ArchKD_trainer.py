# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import os
import torch
import torch.nn as nn
import torchvision

from torch.utils.tensorboard import SummaryWriter

from genotypes.genotypes import parse_dag_to_alpha
import teacher_models
from trainer.searchStage_trainer import SearchStageTrainer_WithSimpleKD
from utils.loss import AlphaArchLoss, KD_Loss, SoftTargetKLLoss
import utils
from utils.data_util import get_data, split_dataloader
from utils.file_management import load_teacher_checkpoint_state
from utils.params_util import collect_params
from utils.eval_util import AverageMeter, accuracy, validate

from utils.data_prefetcher import data_prefetcher

from models.architect import Architect, Architect_Arch
from utils.visualize import showModelOnTensorboard


class SearchStageTrainer_ArchKD(SearchStageTrainer_WithSimpleKD):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def construct_model(self):
        # ================= define data loader ==================
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=self.config.cutout_length, validation=False, advanced=self.config.advanced
        )
        self.train_loader, self.valid_loader = split_dataloader(train_data, self.config.train_portion, self.config.batch_size, self.config.workers)
       
        print("---------- init model ----------")
        # ================= define criteria ==================
        self.hard_criterion = nn.CrossEntropyLoss().to(self.device)
        self.soft_criterion = SoftTargetKLLoss(self.T).to(self.device)
        self.criterion = AlphaArchLoss().to(self.device)
        # ================= Student model ==================
        model = self.Controller(input_size, input_channels, self.config.init_channels, n_classes, self.config.layers, self.criterion, genotype=self.config.genotype, device_ids=self.config.gpus, spec_cell=self.config.spec_cell, slide_window=self.sw)
        self.model = model.to(self.device)
        # ================= Teacher Model ==================
        teacher_model = self.load_teacher(n_classes)
        setattr(teacher_model, "alpha_DAG", parse_dag_to_alpha(teacher_model.DAG, self.model.n_big_nodes, n_ops=4, window=self.model.window, device=self.device))
        self.teacher_model = teacher_model.to(self.device)

        # validate(self.valid_loader, 
        #         self.teacher_model,
        #         self.hard_criterion, 
        #         self.device, 
        #         print_freq=100000,
        #         printer=self.logger.info, 
        #         model_description="{} <- ({})".format(self.config.teacher_name, self.config.teacher_path))
        showModelOnTensorboard(self.writer, self.teacher_model, self.train_loader)

        showModelOnTensorboard(self.writer, self.model, self.train_loader)
        print("---------- init model end! ----------")
        # ================= build Optimizer ==================
        print("---------- get optimizer ----------")
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), self.config.alpha_lr, betas=(0.5, 0.999), weight_decay=self.config.alpha_weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        self.architect = Architect_Arch(self.model, self.teacher_model, self.config.w_momentum, self.config.w_weight_decay)
    
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
    
    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        hard_losses = AverageMeter()
        soft_losses = AverageMeter()
        arch_losses = AverageMeter()
        arch_hard_losses = AverageMeter()
        arch_soft_losses = AverageMeter()
        arch_depth_losses = AverageMeter()

        cur_lr = self.lr_scheduler.get_last_lr()[0]

        # 構造パラメータを表示
        # self.model.print_alphas(self.logger)

        self.model.train()
        self.teacher_model.train()

        prefetcher_trn = data_prefetcher(self.train_loader)
        prefetcher_val = data_prefetcher(self.valid_loader)
        trn_X, trn_y = prefetcher_trn.next()
        val_X, val_y = prefetcher_val.next()
        i = 0
        while trn_X is not None:
            i += 1
            N = trn_X.size(0)
            self.steps += 1

            # ================= optimize architecture parameter ==================
            self.alpha_optim.zero_grad()
            arch_loss = arch_hard_loss = arch_soft_loss = self.architect.unrolled_backward_archkd(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim)
            self.alpha_optim.step()

            self.alpha_optim.zero_grad()
            alpha = self.architect.net.alpha_DAG
            self.n_nodes = self.config.layers // 3
            d_depth1 = self.cal_depth(alpha[0 * self.n_nodes: 1 * self.n_nodes], self.n_nodes, self.sw)
            d_depth2 = self.cal_depth(alpha[1 * self.n_nodes: 2 * self.n_nodes], self.n_nodes, self.sw)
            d_depth3 = self.cal_depth(alpha[2 * self.n_nodes: 3 * self.n_nodes], self.n_nodes, self.sw)
            depth_loss = -self.depth_coef * (d_depth1 + d_depth2 + d_depth3)
            depth_loss.backward()
            self.alpha_optim.step()
            
            # ================= optimize network parameter ==================
            self.w_optim.zero_grad()
            logits = self.model(trn_X)
            
           
            # === KD for optimizing network params ===
            with torch.no_grad():
                teacher_guide = self.teacher_model(trn_X)
            hard_loss = soft_loss = loss = self.model.criterion(self.model.alphas_list(), self.teacher_model.alpha_DAG)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            # 学習過程の記録用
            losses.update(loss.item(), N)
            hard_losses.update(hard_loss.item(), N)
            soft_losses.update(soft_loss.item(), N)
            arch_losses.update(arch_loss.item(), N)
            arch_hard_losses.update(arch_hard_loss.item(), N)
            arch_soft_losses.update(arch_soft_loss.item(), N)
            arch_depth_losses.update(depth_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if i % self.config.print_freq == 0 or i == len(self.train_loader) - 1:
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Hard Loss {hard_losses.val:.4f} ({hard_losses.avg:.4f})\t'
                        f'Soft Loss {soft_losses.val:.4f} ({soft_losses.avg:.4f})\t'
                        f'Arch Loss {arch_losses.val:.4f} ({arch_losses.avg:.4f})\t'
                        f'Arch Hard Loss {arch_hard_losses.val:.4f} ({arch_hard_losses.avg:.4f})\t'
                        f'Arch Soft Loss {arch_soft_losses.val:.4f} ({arch_soft_losses.avg:.4f})\t'
                        f'Arch depth Loss {arch_depth_losses.val:.4f} ({arch_depth_losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                        )
            
            trn_X, trn_y = prefetcher_trn.next()
            val_X, val_y = prefetcher_val.next()
        
        printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, hard_losses.avg, soft_losses.avg, losses.avg, arch_hard_losses.avg, arch_soft_losses.avg, arch_losses.avg, arch_depth_losses.avg

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

                logits = self.model(X)
                loss = self.hard_criterion(logits, y)

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
                
                if i % self.config.print_freq == 0 or i == len(self.valid_loader) - 1:
                    printer(f'Valid: Epoch: [{epoch}][{i}/{len(self.valid_loader)}]\t'
                            f'Step {self.steps}\t'
                            f'Loss {losses.avg:.4f}\t'
                            f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})')
                
                X, y = prefetcher.next()

        self.writer.add_scalar('val/loss', losses.avg, self.steps)
        self.writer.add_scalar('val/top1', top1.avg, self.steps)
        self.writer.add_scalar('val/top5', top5.avg, self.steps)

        printer("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, losses.avg
