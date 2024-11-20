# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter

from genotypes.genotypes import parse_dag_to_alpha
from models.search_stage import SearchStage, SearchStageController, SearchStageController_FullCascade, SearchStageControllerPartialConnection
import teacher_models
from trainer.searchStage_trainer import SearchStageTrainer_WithSimpleKD
from utils.loss import AlphaArchLoss, CosineScheduler, KD_Loss, SoftTargetKLLoss, WeightedCombinedLoss
import utils
from utils.data_util import get_data, split_dataloader
from utils.file_management import load_teacher_checkpoint_state
from utils.params_util import collect_params
from utils.eval_util import AverageMeter, accuracy, validate

from utils.data_prefetcher import data_prefetcher

from models.architect import Architect, Architect_Arch
from utils.visualize import showModelOnTensorboard


class SearchEvaluateStageTrainer_ArchKD(SearchStageTrainer_WithSimpleKD):
    def __init__(self, config) -> None:
        self.config = config

        self.world_size = 1
        self.gpu = self.config.gpus
        self.save_epoch = 1
        self.ckpt_path = self.config.path
        self.device = utils.set_seed_gpu(config.seed, config.gpus)
        # self.Controller = SearchStageControllerPartialConnection if self.config.pcdarts else SearchStageController
        self.sw = self.config.slide_window
        if self.config.pcdarts:
            self.Controller = SearchStageControllerPartialConnection
        elif self.config.cascade:
            self.Controller = SearchStageController_FullCascade
            self.sw = self.config.layers // 3
        else:
            self.Controller = SearchStageController
            
        """get the train parameters"""
        self.eval_epochs = self.config.eval_epochs
        self.search_epochs = self.config.epochs
        self.total_epochs = self.search_epochs + self.eval_epochs
        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.global_batch_size = self.world_size * self.train_batch_size
        self.max_lr = self.config.w_lr * self.world_size

        self.T = self.config.T
        self.l = self.config.l
        self.depth_coef = self.config.depth_coef

        """construct the whole network"""
        self.resume_path = self.config.resume_path

        self.steps = 0
        self.log_step = 10
        self.logger = self.config.logger
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.path, "tb"))
        self.writer.add_text('config', config.as_markdown(), 0)
    
    def construct_model(self):
        # ================= define data loader ==================
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=self.config.cutout_length, validation=False, advanced=self.config.advanced
        )
        self.train_loader, self.valid_loader = split_dataloader(train_data, self.config.train_portion, self.config.batch_size, self.config.workers)
        # ================= Teacher Model ==================
        self.teacher_model = None
        if not self.config.nonkd:
            print("---------- load teacher model ----------")
            teacher_model = self.load_teacher(n_classes)
            setattr(teacher_model, "alpha_DAG", parse_dag_to_alpha(teacher_model.DAG, self.config.layers//3, n_ops=4, window=self.sw, device=self.device))
            self.teacher_model = teacher_model.to(self.device)
            # validate(self.valid_loader, 
            #         self.teacher_model,
            #         self.hard_criterion, 
            #         self.device, 
            #         print_freq=100000,
            #         printer=self.logger.info, 
            #         model_description="{} <- ({})".format(self.config.teacher_name, self.config.teacher_path))
            # showModelOnTensorboard(self.writer, self.teacher_model, self.train_loader)
        print("---------- init student model ----------")
        # ================= define criteria ==================
        self.lossWeight_scheduler = None
        if not self.config.nonkd:
            self.hard_criterion = nn.CrossEntropyLoss().to(self.device)
            self.arch_criterion = AlphaArchLoss(self.teacher_model.alpha_DAG).to(self.device)
            self.loss_functions = [self.hard_criterion, self.arch_criterion]
            self.loss_weights = [1.0, self.config.l]
            self.lossWeight_scheduler = CosineScheduler(initial_value=self.config.l, final_value=self.config.final_l, total_steps=self.total_epochs)
        else:
            self.hard_criterion = nn.CrossEntropyLoss().to(self.device)
            self.loss_functions = [self.hard_criterion]
            self.loss_weights = [1.0]
        self.criterion = WeightedCombinedLoss(functions=self.loss_functions, weights=self.loss_weights).to(self.device)
        # ================= Student model ==================
        model = self.Controller(input_size, input_channels, self.config.init_channels, n_classes, self.config.layers, self.criterion, genotype=self.config.genotype, device_ids=self.config.gpus, spec_cell=self.config.spec_cell, slide_window=self.sw)
        self.model = model.to(self.device)

        # showModelOnTensorboard(self.writer, self.model, self.train_loader)
        print("---------- init student model end! ----------")
        # ================= build Optimizer ==================
        print("---------- get optimizer ----------")
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), self.config.alpha_lr, betas=(0.5, 0.999), weight_decay=self.config.alpha_weight_decay)
       
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.search_epochs, eta_min=self.config.w_lr_min)
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
    
    def freeze_alphaParams(self):
        if self.config.discrete:
            self._discrete_alpha()
        else:
            current_state_dict = self.model.state_dict()
            count = 0
            for name, param in self.model.named_parameters():
                if 'alpha' in name:
                    self.model.alpha_DAG[count] = current_state_dict[name] = F.softmax(param, dim=-1)
                    count += 1
            self.model.load_state_dict(current_state_dict, strict=True)
        
        for name, param in self.model.named_parameters():
            if 'alpha' in name:
                param.requires_grad = False
        self.logger.info(f"--> Loaded alpha parameters are Freezed")

    def _discrete_alpha(self):
        current_DAG = self.model.DAG()
        current_state_dict = self.model.state_dict()
        discrete_alpha_list = parse_dag_to_alpha(current_DAG, n_ops=self.model.alpha_DAG[0][0].size(0), window=self.sw, device=self.device)
        count = 0
        for name, param in self.model.named_parameters():
            if 'alpha' in name:
                self.model.alpha_DAG[count] = current_state_dict[name] = discrete_alpha_list[count]
                count += 1
        self.model.load_state_dict(current_state_dict, strict=True)
        self.logger.info(f"--> Archtecture parameters are DISCRETED")

        return
    
    def reset_model(self, input_size):
        current_alpha = self.model.alpha_DAG

        model = self.Controller(input_size, self.model.net.C_in, self.config.init_channels, self.model.net.n_classes, self.config.layers, self.criterion, genotype=self.config.genotype, device_ids=self.config.gpus, spec_cell=self.config.spec_cell, slide_window=self.sw)
        self.model = model.to(self.device)

        current_state_dict = self.model.state_dict()
        count = 0
        for name, param in self.model.named_parameters():
            if 'alpha' in name:
                self.model.alpha_DAG[count] = current_state_dict[name] = current_alpha[count]
                count += 1
        self.model.load_state_dict(current_state_dict, strict=True)

        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), self.config.alpha_lr, betas=(0.5, 0.999), weight_decay=self.config.alpha_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.eval_epochs, eta_min=self.config.w_lr_min)
        self.architect = Architect_Arch(self.model, self.teacher_model, self.config.w_momentum, self.config.w_weight_decay)
    
        self.logger.info(f"--> Network parameter is reseted.")

    def switch_evaluation(self):
        # ================= re-define data loader ==================
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=16, validation=False, advanced=self.config.advanced
        )
        self.train_loader, self.valid_loader = split_dataloader(train_data, 0.9, self.config.batch_size, self.config.workers)
        if self.config.reset:
            self.reset_model(input_size)
        else:
            self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.eval_epochs, eta_min=self.config.w_lr_min)
        self.freeze_alphaParams()
        self.model.print_alphas(self.logger, fix=True)
        
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.eval_epochs, eta_min=self.config.w_lr_min)


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
        cur_arch_weights = None
        if self.lossWeight_scheduler:
            cur_arch_weights = [self.loss_weights[0], self.lossWeight_scheduler.get_increase_value(current_step=epoch)]

        # 構造パラメータを表示
        # self.model.print_alphas(self.logger)

        self.model.train()
        if not self.config.nonkd:
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
            if epoch < self.search_epochs:
                self.alpha_optim.zero_grad()
                archLosses = self.architect.unrolled_backward_archkd(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim, cur_arch_weights)
                # archLosses = self.architect.unrolled_backward_archkd_updated_weight(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim)
                arch_hard_loss = archLosses[0]
                arch_alphaloss = archLosses[1]
                arch_loss = archLosses[-1]
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
            logits = self.model(trn_X, fix=False if epoch < self.search_epochs else True)
            # teacher_guide = self.teacher_model(trn_X)
           
            hard_loss = soft_loss = loss = self.hard_criterion(logits, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            # 学習過程の記録用
            losses.update(loss.item(), N)
            hard_losses.update(hard_loss.item(), N)
            soft_losses.update(soft_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            if epoch < self.search_epochs:
                arch_losses.update(arch_loss.item(), N)
                arch_hard_losses.update(arch_hard_loss.item(), N)
                arch_soft_losses.update(arch_alphaloss.item(), N)
                arch_depth_losses.update(depth_loss.item(), N)

            if i % self.config.print_freq == 0 or i == len(self.train_loader) - 1:
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Arch Loss {arch_losses.val:.4f} ({arch_losses.avg:.4f})\t'
                        f'Arch Hard Loss {arch_hard_losses.val:.4f} ({arch_hard_losses.avg:.4f})\t'
                        f'Arch Alpha Loss {arch_soft_losses.val:.4f} ({arch_soft_losses.avg:.4f})\t'
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

                logits = self.model(X, fix=False if epoch < self.search_epochs else True)
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