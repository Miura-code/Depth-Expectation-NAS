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
from trainer.searchStage_trainer import SearchStageTrainer, SearchStageTrainer_WithSimpleKD
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
    utils.set_seed_gpu(config.seed, config.gpus)
    
    # ================= define trainer ==================
    trainer = SearchStageTrainer_WithSimpleKD(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch
    # ================= record initial genotype ==================
    previous_arch = macro_arch = trainer.model.DAG()
    DAG_path = os.path.join(config.DAG_path, "EP00")
    plot_path = os.path.join(config.plot_path, "EP00")
    caption = "Initial DAG"
    plot2(macro_arch.DAG1, plot_path + '-DAG1', caption)
    plot2(macro_arch.DAG2, plot_path + '-DAG2', caption)
    plot2(macro_arch.DAG3, plot_path + '-DAG3', caption)
    save_DAG(macro_arch, DAG_path)
    
    # loss, accを格納する配列
    record = RecordDataclass()

    best_top1 = 0.
    for epoch in tqdm(range(start_epoch, trainer.total_epochs)):
        train_top1, train_hardloss, train_softloss, train_loss, arch_train_hardloss, arch_train_softloss, arch_train_loss, arch_depth_loss = trainer.train_epoch(epoch, printer=logger.info)
        val_top1, val_loss = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()

        # ================= record genotype logs ==================
        macro_arch = trainer.model.DAG()
        logger.info("DAG = {}".format(macro_arch))
        
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        DAG_path = os.path.join(config.DAG_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot2(macro_arch.DAG1, plot_path + '-DAG1', caption)
        plot2(macro_arch.DAG2, plot_path + '-DAG2', caption)
        plot2(macro_arch.DAG3, plot_path + '-DAG3', caption)

        # ================= write tensorboard ==================
        trainer.writer.add_scalar('train/lr', round(trainer.lr_scheduler.get_last_lr()[0], 5), epoch)
        trainer.writer.add_scalar('train/hardloss', train_hardloss, epoch)
        trainer.writer.add_scalar('train/softloss', train_softloss, epoch)
        trainer.writer.add_scalar('train/loss', train_loss, epoch)
        trainer.writer.add_scalar('train/archhardloss', arch_train_hardloss, epoch)
        trainer.writer.add_scalar('train/archsoftloss', arch_train_softloss, epoch)
        trainer.writer.add_scalar('train/archloss', arch_train_loss, epoch)
        trainer.writer.add_scalar('train/archdepthloss', arch_depth_loss, epoch)
        trainer.writer.add_scalar('train/top1', train_top1, epoch)
        trainer.writer.add_scalar('val/loss', val_loss, epoch)
        trainer.writer.add_scalar('val/top1', val_top1, epoch)

        # ================= record genotype and checkpoint ==================
        if best_top1 < val_top1:
            best_top1, is_best = val_top1, True
            best_macro = macro_arch
        else:
            is_best = False
        if previous_arch != macro_arch:
            save_DAG(macro_arch, DAG_path, is_best=is_best)
        trainer.save_checkpoint(epoch, is_best=is_best)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Final Best Genotype = {}".format(best_macro))

    png2gif(config.DAG_path, file_name="DAG1_history", pattern="*DAG1*")
    png2gif(config.DAG_path, file_name="DAG2_history", pattern="*DAG2*")
    png2gif(config.DAG_path, file_name="DAG3_history", pattern="*DAG3*")


def main():
    config = SearchStageConfig()
    run_task(config)


if __name__ == "__main__":
    main()
