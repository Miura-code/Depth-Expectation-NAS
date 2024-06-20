# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import utils
from utils.logging_util import get_std_logging
from config.searchStage_config import SearchStageConfig
from trainer.searchStage_trainer import SearchStageTrainer
from trainer.searchShareStage_trainer import SearchShareStageTrainer
from genotypes.genotypes import save_DAG
from utils.visualize import plot2, png2gif
from utils.eval_util import RecordDataclass

from tqdm import tqdm


def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.exp_name)))
    config.logger = logger

    config.print_params(logger.info)
    
    # set seed
    utils.set_seed_gpu(config.seed, config.gpus[0])
    
    if config.share_stage:
        trainer = SearchShareStageTrainer(config)
    else:
        trainer = SearchStageTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch
    
    previous_arch = macro_arch = trainer.model.DAG()
    plot_path = os.path.join(config.DAG_path, "EP00")
    caption = "Initial DAG"
    if config.share_stage:
        plot2(macro_arch.DAG1, plot_path + '-DAG', caption)
    else:
        plot2(macro_arch.DAG1, plot_path + '-DAG1', caption)
        plot2(macro_arch.DAG2, plot_path + '-DAG2', caption)
        plot2(macro_arch.DAG3, plot_path + '-DAG3', caption)
    save_DAG(macro_arch, plot_path)
    
    # loss, accを格納する配列
    record = RecordDataclass()

    best_top1 = 0.
    for epoch in tqdm(range(start_epoch, trainer.total_epochs)):
        train_top1, train_loss = trainer.train_epoch(epoch, printer=logger.info)
        val_top1, val_loss = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()

        # plot macro architecture
        macro_arch = trainer.model.DAG()
        logger.info("DAG = {}".format(macro_arch))

        plot_path = os.path.join(config.DAG_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        if config.share_stage:
            plot2(macro_arch.DAG1, plot_path + '-DAG', caption)
        else:
            plot2(macro_arch.DAG1, plot_path + '-DAG1', caption)
            plot2(macro_arch.DAG2, plot_path + '-DAG2', caption)
            plot2(macro_arch.DAG3, plot_path + '-DAG3', caption)
        if previous_arch != macro_arch:
            save_DAG(macro_arch, plot_path + '-DAG')
        previous_arch = macro_arch

        if best_top1 < val_top1:
            best_top1, is_best = val_top1, True
            best_macro = macro_arch
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best=is_best)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))
        
        record.add(train_loss, val_loss, train_top1, val_top1)
        record.save(config.path)

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Final Best Genotype = {}".format(best_macro))

    if config.share_stage:
        png2gif(config.DAG_path, pattern="*DAG*")
    else:
        png2gif(config.DAG_path, file_name="DAG1_history", pattern="*DAG1*")
        png2gif(config.DAG_path, file_name="DAG2_history", pattern="*DAG2*")
        png2gif(config.DAG_path, file_name="DAG3_history", pattern="*DAG3*")


def main():
    config = SearchStageConfig()
    run_task(config)


if __name__ == "__main__":
    main()
