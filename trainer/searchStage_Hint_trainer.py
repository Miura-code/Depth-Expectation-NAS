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
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from torch.utils.tensorboard import SummaryWriter

from models.regressor import Regressor_Controller
import teacher_models
import utils
from utils.data_util import get_data, split_dataloader
from utils.file_management import load_teacher_checkpoint_state
from utils.loss import HintLoss, KD_Loss, SoftTargetKLLoss
from utils.eval_util import AverageMeter, accuracy, validate

from utils.data_prefetcher import data_prefetcher

from models.search_stage import SearchStageController_Hint
from models.architect import Architect, Architect_Hint
from utils.visualize import showModelOnTensorboard


class SearchStageTrainer_HintKD():
    def __init__(self, config) -> None:
        self.config = config

        self.world_size = 1
        self.gpu = self.config.gpus
        self.save_epoch = 1
        self.ckpt_path = self.config.path
        self.device = utils.set_seed_gpu(config.seed, config.gpus)
        self.sw = self.config.slide_window            
        
        self.Controller = SearchStageController_Hint
            
        """get the train parameters"""
        self.hint1_epochs = self.config.hint_epochs[0]
        self.hint2_epochs = self.config.hint_epochs[1]
        self.total_epochs = self.config.epochs
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

        self.construct_model()
    
    def construct_model(self):
        # ================= define data loader ==================
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=self.config.cutout_length, validation=False, advanced=self.config.advanced
        )
        self.train_loader, self.valid_loader = split_dataloader(train_data, self.config.train_portion, self.config.batch_size, self.config.workers)
       
        print("init model")
        # ================= define criteria ==================
        self.hard_criterion = nn.CrossEntropyLoss().to(self.device)
        self.soft_criterion = SoftTargetKLLoss(self.T).to(self.device)
        self.hint_criterion = HintLoss().to(self.device)
        self.criterion = KD_Loss(self.soft_criterion, self.hard_criterion, self.l, self.config.T)
        # ================= Student model ==================
        model = self.Controller(input_size, input_channels, self.config.init_channels, n_classes, self.config.layers, self.criterion, genotype=self.config.genotype, device_ids=self.config.gpus, spec_cell=self.config.spec_cell, slide_window=self.sw, hint_criterion=self.hint_criterion)
        self.model = model.to(self.device)
        showModelOnTensorboard(self.writer, self.model, self.train_loader)
        # ================= Teacher Model ==================
        teacher_model, self.teacher_feature_extractor = self.load_teacher(n_classes)
        self.teacher_model = teacher_model.to(self.device)
        validate(self.valid_loader, 
                self.teacher_model,
                self.hard_criterion, 
                self.device, 
                print_freq=100000,
                printer=self.logger.info, 
                model_description="{} <- ({})".format(self.config.teacher_name, self.config.teacher_path))
        showModelOnTensorboard(self.writer, self.teacher_model, self.train_loader)        
        # ================= Regressor Model ==================
        info_set = [(14, 28, 128, 64),
                    (7, 14, 256, 128)]
        self.Regressor = Regressor_Controller(num_stages=2, reg_info_set=info_set)
        self.Regressor.to_device(self.device)
        print("init model end!")
        # ================= build Optimizer ==================
        print("get optimizer")
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), self.config.alpha_lr, betas=(0.5, 0.999), weight_decay=self.config.alpha_weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        self.architect = Architect_Hint(self.model, self.teacher_model, self.config.w_momentum, self.config.w_weight_decay, teacher_feature_extractor=self.teacher_feature_extractor, Regressor=self.Regressor)
    
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
        
        return_nodes = teacher_models.__dict__[self.config.teacher_name + "_HINTLAYERS"]
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        
        return model, feature_extractor
    def resume_model(self, reset=False, model_path=None):
        print(f"{self.config.checkpoint_reset}")
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logger.info("--> No loaded checkpoint!")
        elif reset or self.checkpoint_reset:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = 0
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.logger.info(f"--> Loaded checkpoint '{model_path}'(Reseted epoch {self.start_epoch})")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.w_optim.load_state_dict(checkpoint['w_optim'])
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
            self.logger.info(f"--> Loaded checkpoint '{model_path}'(epoch {self.start_epoch})")
        
    def save_checkpoint(self, epoch, is_best=False):
        if epoch % self.save_epoch == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'w_optim': self.w_optim.state_dict(),
                     'alpha_optim': self.alpha_optim.state_dict()
                    }
            if is_best:
                best_filename = os.path.join(self.ckpt_path, 'best.pth.tar')
                torch.save(state, best_filename)
    
    def cal_depth(self, alpha, n_nodes, SW):
        """
        SW: sliding window
        """
        assert len(alpha) == n_nodes, "the length of alpha must be the same as n_nodes"
        d = [0, 0]
        for i in range(n_nodes):
            if i + 2 < SW:
                dd = 0
                for j in range(i + 2):
                    dd += alpha[i][j] * (d[j] + 1)
                dd /= (i + 2)
                d.append(dd)
            else:
                dd = 0
                for s, j in enumerate(range(i + 2 - SW, i + 2)):
                    dd += alpha[i][s] * (d[j] + 1)
                dd /= SW
                d.append(dd)
        return sum(sum(d) / n_nodes)
    
    def train_hint_epoch(self, epoch, printer=print, stage=1):
        top1 = AverageMeter()
        top5 = AverageMeter()
        hint_losses = AverageMeter()
        arch_hint_losses = AverageMeter()
        arch_depth_losses = AverageMeter()

        cur_lr = self.lr_scheduler.get_last_lr()[0]

        self.model.print_alphas(self.logger)
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
            # === KD for optimizing architecture params ===
            arch_hint_loss = self.architect.unrolled_backward_hint(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim, stage=stage)
            self.alpha_optim.step()
            
            # ================= calculate depth loss ==================
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
            student_features_DICT = self.model.extract_features(trn_X)
            student_guided = self.Regressor(student_features_DICT["stage"+str(stage)], stage=stage)

            with torch.no_grad():
                teacher_hint_DICT = self.teacher_feature_extractor(trn_X)

            hint_loss = self.model.hint_criterion(student_guided, teacher_hint_DICT["stage"+str(stage)])
            hint_loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(student_features_DICT["logits"], trn_y, topk=(1, 5))
            # 学習過程の記録用
            hint_losses.update(hint_loss.item(), N)
            # arch_hint_losses.update(arch_hint_loss.item(), N)
            arch_depth_losses.update(depth_loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if i % self.config.print_freq == 0 or i == len(self.train_loader) - 1:
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'HintLoss {hint_losses.val:.4f} ({hint_losses.avg:.4f})\t'
                        f'Arch Loss {arch_hint_losses.val:.4f} ({arch_hint_losses.avg:.4f})\t'
                        f'Arch depth Loss {arch_depth_losses.val:.4f} ({arch_depth_losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                        )
            
            trn_X, trn_y = prefetcher_trn.next()
            val_X, val_y = prefetcher_val.next()
        
        printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, hint_losses.avg, arch_hint_losses.avg, arch_depth_losses.avg
    
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

        self.model.print_alphas(self.logger)
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
            # === KD for optimizing architecture params ===
            arch_hard_loss, arch_soft_loss, arch_loss = self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, cur_lr, self.w_optim)
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
            
            # hard_loss = soft_loss = loss = self.hard_criterion(logits, trn_y)
            # === KD for optimizing network params ===
            with torch.no_grad():
                teacher_guide = self.teacher_model(trn_X)
            hard_loss, soft_loss, loss = self.model.criterion(logits, teacher_guide, trn_y, True)
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
