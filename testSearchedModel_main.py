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
from config.searchStage_config import TestSearchStageConfig
from genotypes.genotypes import save_DAG
from models.search_stage import SearchStageController, SearchStageDistributionBetaController, SearchStageDistributionBetaCurriculumController
import utils
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.data_util import get_data
from utils.logging_util import get_std_logging
from utils.eval_util import AverageMeter, accuracy
from utils.measurement_utils import TimeKeeper
import utils.measurement_utils

from utils.params_util import freeze_alphaParams, resume_alpha_discrete, resume_model
from utils.visualize import plot2


config = TestSearchStageConfig()

device = torch.device("cuda")

logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.save)))
config.logger = logger
config.print_params(logger.info)

writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

def main():
    logger.info("Logger is set - test start")

    # ================= set gpu ==================
    utils.set_seed_gpu(config.seed, config.gpus)

    # ================= get dataset ==================
    input_size, input_channels, n_classes, _, valid_data = get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True, advanced=config.advanced)
    # ================= define dataloader ==================
    n_val = len(valid_data)
    split = int(np.floor(config.train_portion * n_val))
    indices = list(range(n_val))
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    test_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               sampler=test_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    
    criterion = nn.CrossEntropyLoss().to(device)

    # ================= set search tranier ==================
    if config.type == "SearchEvalCurriculum":
        model = SearchStageDistributionBetaCurriculumController(input_size, input_channels, config.init_channels, n_classes, config.layers, criterion, genotype=config.genotype, device_ids=config.gpus, spec_cell=config.spec_cell, slide_window=config.slide_window)
    else:
        model = SearchStageController(input_size, input_channels, config.init_channels, n_classes, config.layers, criterion, genotype=config.genotype, device_ids=config.gpus, spec_cell=config.spec_cell, slide_window=config.slide_window)    # 
    
    model = model.to(device)
    # ================= load checkpoint ==================
    model = resume_model(model, model_path=config.resume_path, device=device)
    model._curri = False

    previous_arch = macro_arch = model.DAG()
    DAG_path = os.path.join(config.path, "dag")
    plot_path = os.path.join(config.path, "plot")
    caption = "Initial DAG"
    plot2(macro_arch.DAG1, plot_path + '-DAG1', caption, concat=macro_arch.DAG1_concat)
    plot2(macro_arch.DAG2, plot_path + '-DAG2', caption, concat=macro_arch.DAG2_concat)
    plot2(macro_arch.DAG3, plot_path + '-DAG3', caption, concat=macro_arch.DAG3_concat)
    save_DAG(macro_arch, DAG_path)

    model.print_alphas(logger, fix=True)

    if config.discrete:
        model = resume_alpha_discrete(model, macro_arch, model_path=config.resume_path, device=device, printer=logger.info, sw=config.slide_window)
    model = freeze_alphaParams(model, printer=logger.info)

    # logger.info("param size = %fMB", utils.measurement_utils.count_parameters_in_MB(model))
    # mac, params = utils.measurement_utils.count_ModelSize_byptflops(model, (3,32,32))
    # logger.info("param size = {}MB, mac = {}".format(params, mac))

    # ================= Test model ==================
    time_keeper = TimeKeeper()
    test_top1, test_top5 = validate(model, test_loader, criterion, device)
    time_keeper.end()
    start, end, diff = time_keeper.print_info()

    logger.info("Test Prec(@1, @5) = ({:.4%}, {:.4%})".format(test_top1, test_top5))
    logger.info("Time to Test = ({}, {}, {})".format(start, end, diff))

    writer.add_text('result', utils.ListToMarkdownTable(["ACC_TOP1", "ACC_TOP5"], [test_top1, test_top5]), 0)
    writer.close()

def validate(model, valid_loader, criterion, device):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in tqdm(enumerate(valid_loader)):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X, fix=True)
            loss = criterion(logits, y)

            prec1, prec5 = accuracy(logits, y, topk=(1,5))
            # ================= record process ==================
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

    return top1.avg, top5.avg


if __name__ == "__main__":
    cudnn.benchmark = True
    main()
