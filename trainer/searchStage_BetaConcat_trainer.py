import os
import torch
import torch.nn as nn

from models.search_stage import SearchStageDistributionBetaController
from trainer.searchStage_trainer import SearchStageTrainer

import utils
import utils.measurement_utils
from utils.loss import CellLength_beta, Expected_Depth_Loss_beta, Lp_loss_beta, WeightedCombinedLoss
from utils.data_util import get_data, split_dataloader
from utils.eval_util import AverageMeter, accuracy
from utils.data_prefetcher import data_prefetcher

from models.architect import Architect


class SearchStageTrainer_BetaConcat(SearchStageTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.Controller = SearchStageDistributionBetaController
    
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
        elif self.config.arch_criterion == "expected":
            self.beta_criterioin = Expected_Depth_Loss_beta(sw=self.sw, n_node=self.config.layers//3, theta=mac_ratio, device=self.device).to(self.device)

        self.loss_functions = [self.hard_criterion, self.beta_criterioin]
        self.loss_weights = [1.0, self.config.g]
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
        self.architect = Architect(self.model, self.config.w_momentum, self.config.w_weight_decay)
        
    
    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        arch_losses = AverageMeter()
        arch_depth_losses = AverageMeter()

        cur_lr = self.lr_scheduler.get_last_lr()[0]

        # 構造パラメータを表示
        # self.model.print_alphas(self.logger)

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
            self.alpha_optim.zero_grad()
            archLosses = self.architect.unrolled_backward_betaConstraint(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim)
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
           
            loss = self.hard_criterion(logits, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            # 学習過程の記録用
            arch_losses.update(arch_loss.item(), N)
            arch_depth_losses.update(depth_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if i % self.config.print_freq == 0 or i == len(self.train_loader) - 1:
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Arch Loss {arch_losses.val:.4f} ({arch_losses.avg:.4f})\t'
                        f'Arch depth Loss {arch_depth_losses.val:.4f} ({arch_depth_losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                        )
            
            trn_X, trn_y = prefetcher_trn.next()
            val_X, val_y = prefetcher_val.next()
        
        printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, losses.avg, arch_losses.avg, arch_depth_losses.avg