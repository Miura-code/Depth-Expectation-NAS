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
from tqdm import tqdm
from models.augment_stage import EvaluateRelaxedStageController
from trainer.searchStage_trainer import SearchStageTrainer_WithSimpleKD
from utils.eval_util import validate
import utils
from utils.data_util import get_data, split_dataloader
from utils.file_management import load_teacher_checkpoint_state
from utils.eval_util import AverageMeter, accuracy
from utils.loss import KD_Loss, SoftTargetKLLoss
from utils.visualize import showModelOnTensorboard

class EvaluateRelaxedStageTrainer(SearchStageTrainer_WithSimpleKD):
    def __init__(self, config) -> None:
        self.config = config

        self.world_size = 1
        self.gpu = self.config.gpus
        self.save_epoch = 1
        self.ckpt_path = self.config.path
        self.device = utils.set_seed_gpu(config.seed, config.gpus)
        self.sw = self.config.slide_window
        self.Controller = EvaluateRelaxedStageController
            
        """get the train parameters"""
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
        input_size, input_channels, n_classes, train_data, valid_data = get_data(
            self.config.dataset, self.config.data_path, self.config.cutout_length, validation=True, advanced=self.config.advanced
        )
        self.train_loader, self.valid_loader = split_dataloader(train_data, self.config.train_portion, self.config.batch_size, self.config.workers)
        
        print("init model")
        # ================= define criteria ==================
        self.hard_criterion = nn.CrossEntropyLoss().to(self.device)
        self.soft_criterion = SoftTargetKLLoss(self.T).to(self.device)
        self.criterion = KD_Loss(self.soft_criterion, self.hard_criterion, self.l, self.config.T)
        self.use_aux = self.config.aux_weight > 0.
        # ================= Student model ==================
        model = self.Controller(input_size, input_channels, self.config.init_channels, n_classes, self.config.layers, self.criterion, genotype=self.config.genotype, device_ids=self.config.gpus, spec_cell=self.config.spec_cell, slide_window=self.sw, auxiliary=self.use_aux)
        self.model = model.to(self.device)
        # ================= Teacher Model ==================
        if not self.config.nonkd:
            teacher_model = self.load_teacher(n_classes)
            self.teacher_model = teacher_model.to(self.device)

            validate(self.valid_loader, 
                    self.teacher_model,
                    self.hard_criterion, 
                    self.device, 
                    print_freq=100000,
                    printer=self.logger.info, 
                    model_description="{} <- ({})".format(self.config.teacher_name, self.config.teacher_path))
            showModelOnTensorboard(self.writer, self.teacher_model, self.train_loader)
        else:
            self.teacher_model = None

        showModelOnTensorboard(self.writer, self.model, self.train_loader)
        print("init model end!")

        # ================= build Optimizer ==================
        print("get optimizer")
        self.w_optim = torch.optim.SGD(self.model.weights(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        self.alpha_optim = None

    def resume_alpha(self, reset=False, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logger.info("--> No loaded checkpoint!")
        elif reset or self.config.checkpoint_reset:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            current_state_dict = self.model.state_dict()
            for name, param in self.model.named_parameters():
                if 'alpha' in name:
                    if name in checkpoint['model']:
                        current_state_dict[name] = checkpoint['model'][name]
            self.model.load_state_dict(current_state_dict, strict=True)

            self.start_epoch = 0
            self.logger.info(f"--> Loaded checkpoint '{model_path}'(Reseted epoch {self.start_epoch})")
        else:
            model_path = model_path or self.resume_path
            checkpoint = torch.load(model_path, map_location=self.device)

            current_state_dict = self.model.state_dict()
            for name, param in self.model.named_parameters():
                if 'alpha' in name:
                    if name in checkpoint['model']:
                        current_state_dict[name] = checkpoint['model'][name]
            self.model.load_state_dict(current_state_dict, strict=True)

            self.start_epoch = checkpoint['epoch']
            self.steps = checkpoint['steps']
            self.w_optim.load_state_dict(checkpoint['w_optim'])
            self.logger.info(f"--> Loaded checkpoint '{model_path}'(epoch {self.start_epoch})")

        self.freeze_alphaParams()
    
    def freeze_alphaParams(self):
        for name, param in self.model.named_parameters():
            if 'alpha' in name:
                param.requires_grad = False
        self.logger.info(f"--> Loaded alpha parameters are Freezed")
    
    def save_checkpoint(self, epoch, is_best=False):
        if epoch % self.save_epoch == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'steps': self.steps,
                     'model': self.model.state_dict(),
                     'w_optim': self.w_optim.state_dict(),
                    }
            if is_best:
                best_filename = os.path.join(self.ckpt_path, 'best.pth.tar')
                torch.save(state, best_filename)
    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        hard_losses = AverageMeter()
        soft_losses = AverageMeter()

        cur_lr = self.lr_scheduler.get_last_lr()[0]

        self.model.print_alphas(self.logger)
        self.model.train()
        if not self.config.nonkd:
            self.teacher_model.train()

        i = 0
        for X, y in tqdm(self.train_loader):
            i += 1
            N = X.size(0)
            self.steps += 1

            X = X.to(self.device)
            y = y.to(self.device)

            # ================= optimize network parameter ==================
            self.w_optim.zero_grad()
            logits, aux_logits = self.model(X)
            # hard_loss = soft_loss = loss = self.hard_criterion(logits, trn_y)
            if self.config.nonkd:
                # === (Not KD for optimizing network params) ===
                hard_loss = soft_loss = loss = self.hard_criterion(logits, y)
            else:
                # === KD for optimizing network params ===
                with torch.no_grad():
                    teacher_guide = self.teacher_model(X)
                hard_loss, soft_loss, loss = self.model.criterion(logits, teacher_guide, y, True)
            if self.use_aux:
                loss += self.config.aux_weight * self.hard_criterion(aux_logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.w_grad_clip)
            self.w_optim.step()

            prec1, prec5 = accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if (i % self.config.print_freq == 0 or i == len(self.train_loader) - 1):
                printer(f'Train: Epoch: [{epoch}][{i}/{len(self.train_loader) - 1}]\t'
                        f'Step {self.steps}\t'
                        f'lr {round(cur_lr, 5)}\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Hard Loss {hard_losses.val:.4f} ({hard_losses.avg:.4f})\t'
                        f'Soft Loss {soft_losses.val:.4f} ({soft_losses.avg:.4f})\t'
                        f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})\t'
                    )

        printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))

        return top1.avg, hard_losses.avg, soft_losses.avg, losses.avg
    
    def val_epoch(self, epoch, printer):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        self.model.eval()

        i = 0

        with torch.no_grad():
            for X, y in tqdm(self.valid_loader):
                N = X.size(0)
                i += 1

                X = X.to(self.device)
                y = y.to(self.device)

                logits, _ = self.model(X)
                loss = self.hard_criterion(logits, y)

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
                
                # if (i % self.config.print_freq == 0 or i == len(self.valid_loader) - 1):
                #     printer(f'Valid: Epoch: [{epoch}][{i}/{len(self.valid_loader)}]\t'
                #             f'Step {self.steps}\t'
                #             f'Loss {losses.avg:.4f}\t'
                #             f'Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})')
                                
            printer("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, losses.avg

