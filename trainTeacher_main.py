
'''
Fine tune teacher model for knowledge distillation using timm
'''
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from utils.data_prefetcher import data_prefetcher
from utils.data_util import get_data
from utils.eval_util import AverageMeter, RecordDataclass, accuracy
from utils.file_management import save_checkpoint
from utils.logging_util import get_std_logging
from timm_.models import create_model
from timm.models import create_model as timm_create_model

from utils.parser import BaseConfig, get_parser, parse_gpus
from utils.visualize import showModelOnTensorboard

LOSS_TYPES = ["training_loss", "validation_loss"]
ACC_TYPES = ["training_accuracy", "validation_accuracy"]

class Config(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search cells of H-DAS config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=200, help='# of training epochs')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
        parser.add_argument('--model_name', type=str, default='densenet121', help='teacher model name')
        parser.add_argument('--cutout_length', type=int, default=0, help='cutout length')
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')

        return parser
    
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = '../data/'
        
        self.path = os.path.join(f'results/teacher/{self.dataset}/{self.model_name}', self.name)
        self.exp_name = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        self.path = os.path.join(self.path, self.exp_name)
        
        self.gpus = parse_gpus(self.gpus)

def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger
    config.print_params(logger.info)
    
    # set seed
    device = utils.set_seed_gpu(config.seed, config.gpus)

    # ================= define data loader ==================
    input_size, input_channels, n_classes, train_data = get_data(
        config.dataset, config.data_path, cutout_length=config.cutout_length, validation=False, advanced=True
    )
    n_train = len(train_data)
    split = int(np.floor(config.train_portion * n_train))
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=config.batch_size,
                                                    sampler=train_sampler,
                                                    num_workers=config.workers,
                                                    pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=config.batch_size,
                                                    sampler=valid_sampler,
                                                    num_workers=config.workers,
                                                    pin_memory=True)
    # ================= load model from timm ==================
    try:
        model = create_model(config.model_name, pretrained=False, num_classes=n_classes)
    except RuntimeError as e:
        model = timm_create_model(config.model_name, pretrained=False, num_classes=n_classes)

    model = model.to(device)

    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)
    showModelOnTensorboard(writer, model, train_loader)
    print("load model end!")
    # ================= build Optimizer ==================
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(config.optim, config.epochs, eta_min=config.lr_min)
    milestone = [int(0.5*config.epochs), int(0.75*config.epochs)]
    gamma = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestone, gamma=gamma)
    # ================= define criteria ==================
    criterion = nn.CrossEntropyLoss().to(device)
    
    # loss, accを格納する配列
    Record = RecordDataclass(LOSS_TYPES, ACC_TYPES)

    best_top1 = 0.
    steps = 0
    for epoch in tqdm(range(0, config.epochs)):
        train_top1, train_loss, steps = train(epoch, config.epochs, steps, model, train_loader, optimizer, criterion, printer=logger.info)
        val_top1, val_loss = valid(epoch, config.epochs, model, valid_loader, criterion, printer=logger.info)
        lr_scheduler.step()

        writer.add_scalar('train/lr', round(lr_scheduler.get_last_lr()[0], 5), epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/top1', train_top1, epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/top1', val_top1, epoch)

    
        if best_top1 < val_top1:
            best_top1, is_best = val_top1, True
        else:
            is_best = False
        model_state = {'config': config,
                     'epoch': epoch,
                     'steps': steps,
                     'model': model.state_dict(),
                     'w_optim': optimizer.state_dict(),
        }
        save_checkpoint(model_state, config.path, is_best=is_best)

        Record.add(LOSS_TYPES+ACC_TYPES, [train_loss, val_loss, train_top1, val_top1])
        Record.save(config.path)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))
    
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    writer.add_text('result/acc', utils.ListToMarkdownTable(["best_val_acc"], [best_top1]), 0)

def train(epoch, total_epoch, step, model, train_loader, optimizer, criterion, printer):
    model.train()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    prefetcher_trn = data_prefetcher(train_loader)
    trn_X, trn_y = prefetcher_trn.next()
    while trn_X is not None:
        N = trn_X.size(0)
        step += 1
        
        # ================= optimize network parameter ==================
        optimizer.zero_grad()
        logits = model(trn_X)
        loss = criterion(logits, trn_y)
        loss.backward()
        optimizer.step()
        # ================= evaluate model ==================
        prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)
                
        trn_X, trn_y = prefetcher_trn.next()

    printer("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, total_epoch - 1, top1.avg))

    return top1.avg, losses.avg, step

def valid(epoch, total_epochs, model, valid_loader, criterion, printer):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()
    prefetcher = data_prefetcher(valid_loader)
    X, y = prefetcher.next()
    i = 0

    with torch.no_grad():
        while X is not None:
            N = X.size(0)
            i += 1

            logits = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)
            
            X, y = prefetcher.next()

    printer("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch, total_epochs - 1, top1.avg))
    
    return top1.avg, losses.avg

def main():
    config = Config()
    run_task(config)


if __name__ == "__main__":
    main()
