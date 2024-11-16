import os
import torch
import torch.nn as nn

from models.search_stage import SearchStageDistributionBetaCurriculumController
from trainer.SearchEvalStage_ArchKD_trainer import SearchEvaluateStageTrainer_ArchKD
from trainer.searchStage_trainer import SearchStageTrainer_WithSimpleKD

import utils
import utils.measurement_utils
from utils.loss import CellLength_beta, CosineScheduler, Lp_loss_beta, WeightedCombinedLoss
from utils.data_util import get_data, split_dataloader
from utils.eval_util import AverageMeter, accuracy
from utils.data_prefetcher import data_prefetcher

from models.architect import Architect_Arch


class SearchEvalStageTrainer_Curriculum(SearchEvaluateStageTrainer_ArchKD, SearchStageTrainer_WithSimpleKD):
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.Controller = SearchStageDistributionBetaCurriculumController
        
        self.eval_epochs = self.config.eval_epochs
        self.curri_epochs = self.config.curriculum_epochs
        self.curriculum_counter = 0
        self.search_epochs = sum(self.curri_epochs)
        self.total_epochs = self.search_epochs + self.eval_epochs
    
    def construct_model(self):
        # ================= define data loader ==================
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=self.config.cutout_length, validation=False, advanced=self.config.advanced
        )
        self.train_loader, self.valid_loader = split_dataloader(train_data, self.config.train_portion, self.config.batch_size, self.config.workers)
        print("---------- init student model ----------")
        # ================= define criteria ==================
        mac_ratio = torch.tensor(
            utils.measurement_utils.MACs_float_to_ratio(self.config.stage_macs)
        )
        self.hard_criterion = nn.CrossEntropyLoss().to(self.device)
        if self.config.arch_criterion == "l1":
            self.beta_criterioin = Lp_loss_beta(self.config.layers//3, mac_ratio).to(self.device)
        elif self.config.arch_criterion == "length":
            self.beta_criterioin = CellLength_beta(self.config.layers//3, mac_ratio).to(self.device)

        
        self.loss_functions = [self.hard_criterion, self.beta_criterioin]
        # カリキュラム学習のため最初は重みgammaを0とする
        self.loss_weights = [1.0, 0.0]
        self.criterion = WeightedCombinedLoss(functions=self.loss_functions, weights=self.loss_weights).to(self.device)
        # ================= Student model ==================
        model = self.Controller(input_size, input_channels, self.config.init_channels, n_classes, self.config.layers, self.criterion, genotype=self.config.genotype, device_ids=self.config.gpus, spec_cell=self.config.spec_cell, slide_window=self.sw)
        self.model = model.to(self.device)
        # showModelOnTensorboard(self.writer, self.model, self.train_loader)
        print("---------- init student model end! ----------")
        # mac, params = utils.measurement_utils.count_ModelSize_byptflops(model, (input_channels,input_size,input_size), path=os.path.join(self.ckpt_path, "model_flops_info.txt"))
        # ================= build Optimizer ==================
        print("---------- get optimizer ----------")
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.alpha_optim = torch.optim.Adam(self.model.archparams(), self.config.alpha_lr, betas=(0.5, 0.999), weight_decay=self.config.alpha_weight_decay)
       
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        self.architect = Architect_Arch(self.model, None, self.config.w_momentum, self.config.w_weight_decay)
            
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

        if epoch == self.curri_epochs[self.curriculum_counter]+1:
            printer("--> Curriculum B part begins")
            self.model._curri = False
            self.loss_weights = [self.loss_weights[0], self.config.g]
            self.curriculum_counter += 1
        
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
            
            # ================= optimize architecture parameter ==================
            if epoch < self.search_epochs:
                self.alpha_optim.zero_grad()
                archLosses = self.architect.unrolled_backward_betaConstraint(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim, self.loss_weights)
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
            logits = self.model(trn_X)
           
            hard_loss = soft_loss = loss = self.hard_criterion(logits, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            # 学習過程の記録用
            losses.update(loss.item(), N)
            hard_losses.update(hard_loss.item(), N)
            soft_losses.update(soft_loss.item(), N)
            if epoch < self.search_epochs:
                arch_losses.update(arch_loss.item(), N)
                arch_hard_losses.update(arch_hard_loss.item(), N)
                arch_soft_losses.update(arch_alphaloss.item(), N)
                arch_depth_losses.update(depth_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if i % self.config.print_freq == 0 or i == len(self.train_loader) - 1:
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Arch Loss {arch_losses.val:.4f} ({arch_losses.avg:.4f})\t'
                        f'Arch Hard Loss {arch_hard_losses.val:.4f} ({arch_hard_losses.avg:.4f})\t'
                        f'Arch Beta Loss {arch_soft_losses.val:.4f} ({arch_soft_losses.avg:.4f})\t'
                        f'Arch depth Loss {arch_depth_losses.val:.4f} ({arch_depth_losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                        )
            
            trn_X, trn_y = prefetcher_trn.next()
            val_X, val_y = prefetcher_val.next()
        
        printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, hard_losses.avg, soft_losses.avg, losses.avg, arch_hard_losses.avg, arch_soft_losses.avg, arch_losses.avg, arch_depth_losses.avg