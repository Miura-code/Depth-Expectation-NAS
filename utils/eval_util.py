# -*- coding: utf-8 -*-
# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)

import csv
import dataclasses
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)
    
    def __str__(self):
        if self.count == 0:
            return str(self.val)
        
        return f'{self.val:.4f} ({self.avg:.4f})'

@dataclasses.dataclass
class RecordDataclass:
  def __init__(self, loss_types:list, acc_types:list):
    self._init_record(loss_types + acc_types)
    self.loss_types = loss_types
    self.acc_types = acc_types

  def _init_record(self, types):
    self.records = {}
    for ty in types:
       self.records[ty] = []

  def add(self, types:list, datas:list):
    """
    Args:
      types: str type list for adding datas to record
      datas: list type list(like [[],[],[]]). This must be same order with types variables
    """
    for j, ty in enumerate(types):
       self.records[ty] += [datas[j]]

  def save(self, path):
    """ 各リストをCSVで保存する """
    with open(path+'/history.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      for ty in self.records.keys():
         writer.writerow(self.records[ty])

    self._plot(path)
    
  def _plot(self, path):
    fig1, ax1 = plt.subplots()
    for ty in self.loss_types:
      ax1.plot(self.records[ty], label=ty)
    ax1.set_title("loss")
    ax1.legend()
    fig1.savefig(path + "/history_loss.png")
    plt.close()

    fig2, ax2 = plt.subplots()
    for ty in self.acc_types:
      ax2.plot(self.records[ty], label=ty)
    ax2.set_title("accuracy")
    ax2.legend()
    fig2.savefig(path + "/history_accuracy.png")
    plt.close()

  def _len(self):
    return  len(self.records[self.records.keys()[0]])

def validate(valid_loader, model, criterion, device, print_freq=50, printer=print, model_description=""):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in tqdm(enumerate(valid_loader)):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = accuracy(logits, y, topk=(1,5))
            # ================= record process ==================
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % print_freq == 0 or step == len(valid_loader) - 1:
                printer(
                    "Test: Step {:03d}/{:03d} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        step, len(valid_loader) - 1, top1=top1, top5=top5))
                
    printer("Teacher=({}) achives Test Prec(@1, @5) = ({:.4%}, {:.4%})".format(model_description, top1.avg, top5.avg))

    return top1.avg, top5.avg