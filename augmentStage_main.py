# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

""" Training augmented macro-architecture(stage) model """
import os
import torch
import torch.nn as nn
import numpy as np
import utils
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.data_util import get_data
from utils.logging_util import get_std_logging
from utils.eval_util import AverageMeter, accuracy
from utils.eval_util import RecordDataclass
from models.augment_stage import AugmentStage
from config.augmentStage_config import AugmentStageConfig


config = AugmentStageConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.exp_name)))
config.logger = logger

config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    utils.set_seed_gpu(config.seed, config.gpus[0])

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True)

    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = config.aux_weight > 0.
    model = AugmentStage(input_size, input_channels, config.init_channels, n_classes, config.layers,
                         use_aux, config.genotype, config.DAG)

    model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # データ数を減らす
    n_train = len(train_data)
    split = int(np.floor(config.train_portion * n_train))
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    n_val = len(valid_data)
    split = int(np.floor(config.train_portion * n_val))
    indices = list(range(n_val))
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               sampler=val_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    # loss, accを格納する配列
    record = RecordDataclass()

    best_top1 = 0.
    # training loop
    for epoch in tqdm(range(config.epochs)):
        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train_top1, train_loss = train(train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        val_top1, val_loss = validate(valid_loader, model, criterion, epoch, cur_step)

        # save
        if best_top1 < val_top1:
            best_top1 = val_top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

        print("")
        logger.info("until now best Prec@1 = {:.4%}".format(best_top1))

        record.add(train_loss, val_loss, train_top1, val_top1)
        record.save(config.path)

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_loader, model, optimizer, criterion, epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(tqdm(train_loader)):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    
    return top1.avg, losses.avg


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = accuracy(logits, y, topk=(1,5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg, losses.avg


if __name__ == "__main__":
    cudnn.benchmark = True
    main()
