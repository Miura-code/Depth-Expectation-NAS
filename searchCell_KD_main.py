# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

'''
H^c-DAS
search specific cells of different stages.
'''
import os
from config.searchCell_config import SearchCellConfig
from genotypes.genotypes import save_DAG
from trainer.searchCell_trainer import SearchCellTrainer_WithSimpleKD
import utils
from utils.eval_util import RecordDataclass
from utils.logging_util import get_std_logging
from utils.visualize import plot, plot2, png2gif

LOSS_TYPES = ["training_hard_loss", "training_soft_loss", "training_loss", "validation_loss"]
ACC_TYPES = ["training_accuracy", "validation_accuracy"]

def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger

    config.print_params(logger.info)
    
    # set seed
    utils.set_seed_gpu(config.seed, config.gpus[0])
    # ================= define trainer ==================
    trainer = SearchCellTrainer_WithSimpleKD(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch
    # ================= record initial genotype ==================
    previous_arch = genotype = trainer.model.genotype()
    plot_path = os.path.join(config.DAG_path, "EP00")
    caption = "Initial genotype"
    
    plot2(genotype.normal1, plot_path + '-normal1', caption)
    plot2(genotype.reduce1, plot_path + '-reduce1', caption)
    plot2(genotype.normal2, plot_path + '-normal2', caption)
    plot2(genotype.reduce2, plot_path + '-reduce2', caption)
    plot2(genotype.normal3, plot_path + '-normal3', caption)
    save_DAG(genotype, plot_path)

    # ================= start training ==================
    # loss, accを格納する配列
    Record = RecordDataclass(LOSS_TYPES, ACC_TYPES)
    best_top1 = 0.
    for epoch in range(start_epoch, trainer.total_epochs):
        train_top1, train_hardloss, train_softloss, train_loss, arch_train_hardloss, arch_train_softloss, arch_train_loss = trainer.train_epoch(epoch, printer=logger.info)
        val_top1, val_loss = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()
        
        # ================= record genotype logs ==================
        genotype = trainer.model.genotype()
        logger.info("genotype = {}".format(genotype))
        
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot2(genotype.normal1, plot_path + '-normal1', caption)
        plot2(genotype.reduce1, plot_path + '-reduce1', caption)
        plot2(genotype.normal2, plot_path + '-normal2', caption)
        plot2(genotype.reduce2, plot_path + '-reduce2', caption)
        plot2(genotype.normal3, plot_path + '-normal3', caption)

        # ================= write tensorboard ==================
        trainer.writer.add_scalar('train/lr', round(trainer.lr_scheduler.get_last_lr()[0], 5), epoch)
        trainer.writer.add_scalar('train/hardloss', train_hardloss, epoch)
        trainer.writer.add_scalar('train/softloss', train_softloss, epoch)
        trainer.writer.add_scalar('train/loss', train_loss, epoch)
        trainer.writer.add_scalar('train/archhardloss', arch_train_hardloss, epoch)
        trainer.writer.add_scalar('train/archsoftloss', arch_train_softloss, epoch)
        trainer.writer.add_scalar('train/archloss', arch_train_loss, epoch)
        trainer.writer.add_scalar('train/top1', train_top1, epoch)
        # trainer.writer.add_scalar('train/top5', prec5.item(), epoch)
        trainer.writer.add_scalar('val/loss', val_loss, epoch)
        trainer.writer.add_scalar('val/top1', val_top1, epoch)
        # trainer.writer.add_scalar('val/top5', top5.avg, epoch)

        # ================= record genotype and checkpoint ==================
        if previous_arch != genotype:
            save_DAG(genotype, plot_path + '-DAG')
        if best_top1 < val_top1:
            best_top1, is_best = val_top1, True
            best_genotype = genotype
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best=is_best)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))
    
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Final Best Genotype = {}".format(best_genotype))
    
    Record.add(LOSS_TYPES+ACC_TYPES, [train_hardloss, train_softloss, train_loss, val_loss, train_top1, val_top1])
    Record.save(config.path)
    
    png2gif(config.DAG_path, file_name="normal1_history", pattern="*normal1*")
    png2gif(config.DAG_path, file_name="reduce1_history", pattern="*reduce1*")
    png2gif(config.DAG_path, file_name="normal2_history", pattern="*normal2*")
    png2gif(config.DAG_path, file_name="reduce2_history", pattern="*reduce2*")
    png2gif(config.DAG_path, file_name="normal3_history", pattern="*normal3*")

    trainer.writer.add_text('result/acc', utils.ListToMarkdownTable(["best_val_acc"], [best_top1]), 0)


def main():
    config = SearchCellConfig()
    run_task(config)


if __name__ == "__main__":
    main()
