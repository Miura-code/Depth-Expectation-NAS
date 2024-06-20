# -*- coding: utf-8 -*-
# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
import csv
import dataclasses
import matplotlib.pyplot as plt
import torch


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
  train_loss = ["train loss"]
  valid_loss = ["valid loss"]
  train_acc = ["train acc"]
  valid_acc = ["valid acc"]

  def add(self, train_loss, valid_loss, train_acc, valid_acc):
    self.train_loss += [train_loss]
    self.valid_loss += [valid_loss]
    self.train_acc += [train_acc]
    self.valid_acc += [valid_acc]


  def save(self, path):
    """ 各リストをCSVで保存する """
    with open(path+'/history.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(self.train_loss)
      writer.writerow(self.valid_loss)
      writer.writerow(self.train_acc)
      writer.writerow(self.valid_acc)

    self._plot(path)
    
  def _plot(self, path):
    fig1, ax1 = plt.subplots()
    ax1.plot(self.train_loss[1:], label="training")
    ax1.plot(self.valid_loss[1:], label="validation")
    ax1.set_title("loss")
    ax1.legend()
    fig1.savefig(path + "/history_loss.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(self.train_acc[1:], label="training")
    ax2.plot(self.valid_acc[1:], label="validation")
    ax2.set_title("accuracy")
    ax2.legend()
    fig2.savefig(path + "/history_accuracy.png")


  def _len(self):
    return  len(self.train_loss)
