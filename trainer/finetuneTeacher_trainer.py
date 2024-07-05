import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.data_util import get_data
from utils.params_util import collect_params
from utils.eval_util import AverageMeter, accuracy
from utils.data_prefetcher import data_prefetcher
from utils.visualize import showModelOnTensorboard
from timm_.models import create_model, resume_checkpoint
from timm.models import create_model as timm_create_model

from torchvision.models import densenet121

class TrainTeacherTrainer():
    def __init__(self, config):
        self.config = config

        self.world_size = 1
        self.gpu = self.config.local_rank
        self.save_epoch = 1
        self.ckpt_path = self.config.path

        """get the train parameters"""
        self.total_epochs = self.config.epochs
        self.train_batch_size = self.config.batch_size
        self.val_batch_size = self.config.batch_size
        self.global_batch_size = self.world_size * self.train_batch_size
        self.max_lr = self.config.w_lr * self.world_size

        """construct the whole network"""
        self.resume_path = self.config.resume_path
        if torch.cuda.is_available():
            # self.device = torch.device(f'cuda:{self.gpu}')
            # torch.cuda.set_device(self.device)
            torch.cuda.set_device(self.config.gpus[0])
            # cudnn.benchmark = True
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.steps = 0
        self.log_step = 10
        self.logger = self.config.logger
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.path, "tb"))
        self.writer.add_text('config', config.as_markdown(), 0)

        self.construct_model()

    
    def construct_model(self):
        # ================= define data loader ==================
        input_size, input_channels, n_classes, train_data = get_data(
            self.config.dataset, self.config.data_path, cutout_length=self.config.cutout_length, validation=False
        )

        n_train = len(train_data)
        split = int(np.floor(self.config.train_portion * n_train))
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
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
        
        print("load pretrained model")
        # ================= define criteria ==================
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # ================= load model from timm ==================
        try:
            model = create_model(self.config.model_name, pretrained=False, num_classes=n_classes)
            # model = densenet121()
        except RuntimeError as e:
            model = timm_create_model(self.config.model_name, pretrained=True, num_classes=n_classes)
        # Do not freeze model
        # self.freeze_model(model)

        self.model = model.to(self.device)
        showModelOnTensorboard(self.writer, self.model, self.train_loader)
        print("load model end!")

        # ================= build Optimizer ==================
        print("get optimizer")
        self.w_optim = torch.optim.SGD(self.model.parameters(), self.config.w_lr, momentum=self.config.w_momentum, weight_decay=self.config.w_weight_decay)

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, self.total_epochs, eta_min=self.config.w_lr_min)
        milestone = [int(0.15*self.total_epochs), int(0.25*self.total_epochs), int(0.5*self.total_epochs), int(0.75*self.total_epochs)]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.w_optim, milestones=milestone, gamma=0.5)


    def freeze_model(self, model):
        """
            freeze the model parameter without last classification layer
        """
        classifier_layer = model.get_classifier()
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in classifier_layer.named_parameters():
            param.requires_grad = True
    
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
                    }
            if is_best:
                best_filename = os.path.join(self.ckpt_path, 'best.pth.tar')
                torch.save(state, best_filename)
    
    def train_epoch(self, epoch, printer=print):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_lr = self.lr_scheduler.get_last_lr()[0]

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
            # ================= optimize network parameter ==================
            self.w_optim.zero_grad()
            logits = self.model(trn_X)
            loss = self.criterion(logits, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.w_grad_clip)
            self.w_optim.step()
            # ================= evaluate model ==================
            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            # if self.steps % self.log_step == 0:
            #     self.writer.add_scalar('train/lr', round(cur_lr, 5), self.steps)
            #     self.writer.add_scalar('train/loss', loss.item(), self.steps)
            #     self.writer.add_scalar('train/top1', prec1.item(), self.steps)
            #     self.writer.add_scalar('train/top5', prec5.item(), self.steps)

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
                loss = self.criterion(logits, y)

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

        printer("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, self.total_epochs - 1, top1.avg))
        
        return top1.avg, losses.avg