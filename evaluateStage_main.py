'''
H^c-DAS
search specific cells of different stages.
'''
import os

from config.evaluateStage_config import EvaluateStageConfig
from genotypes.genotypes import save_DAG
from trainer.evaluateStage_KD_trainer import EvaluateStageTrainer_WithSimpleKD
from trainer.evaluateStage_trainer import EvaluateStageTrainer
import utils
from utils.eval_util import RecordDataclass
from utils.logging_util import get_std_logging

LOSS_TYPES = ["training_loss", "validation_loss"]
LOSS_TYPES_KD = ["training_hard_loss", "training_soft_loss", "training_loss", "validation_loss"]
ACC_TYPES = ["training_accuracy", "validation_accuracy"]

def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger

    config.print_params(logger.info)
    
    # ================= define trainer ==================
    if config.nonkd:
        trainer = EvaluateStageTrainer(config)
        Record = RecordDataclass(LOSS_TYPES, ACC_TYPES)
    else:
        trainer = EvaluateStageTrainer_WithSimpleKD(config)
        Record = RecordDataclass(LOSS_TYPES_KD, ACC_TYPES)
    trainer.construct_model()
    trainer.resume_model()
    start_epoch = trainer.start_epoch
    
    # ================= start training ==================
    best_top1 = 0.
    for epoch in range(start_epoch, trainer.total_epochs):
        drop_prob = config.drop_path_prob * epoch / config.epochs
        trainer.model.drop_path_prob(drop_prob)

        if config.nonkd:
            train_top1, train_loss = trainer.train_epoch(epoch, printer=logger.info)
        else:
            train_top1, train_hardloss, train_softloss, train_loss = trainer.train_epoch(epoch, printer=logger.info)
        val_top1, val_loss = trainer.val_epoch(epoch, printer=logger.info)

        # ================= write tensorboard and record data ==================
        trainer.writer.add_scalar('train/lr', round(trainer.lr_scheduler.get_last_lr()[0], 5), epoch)
        trainer.writer.add_scalar('train/loss', train_loss, epoch)
        trainer.writer.add_scalar('train/top1', train_top1, epoch)
        trainer.writer.add_scalar('val/loss', val_loss, epoch)
        trainer.writer.add_scalar('val/top1', val_top1, epoch)
        if config.nonkd:
            Record.add(LOSS_TYPES+ACC_TYPES, [train_loss, val_loss, train_top1, val_top1])
        else:
            trainer.writer.add_scalar('train/hardloss', train_hardloss, epoch)
            trainer.writer.add_scalar('train/softloss', train_softloss, epoch)
            Record.add(LOSS_TYPES_KD+ACC_TYPES, [train_hardloss, train_softloss, train_loss, val_loss, train_top1, val_top1])
        Record.save(config.path)
        # ================= save checkpoint ==================
        if best_top1 < val_top1:
            best_top1 = val_top1
            is_best = True
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best=is_best)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))

        trainer.lr_scheduler.step()
    
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))

    trainer.writer.add_text('result/acc', utils.ListToMarkdownTable(["best_val_acc"], [best_top1]), 0)


def main():
    config = EvaluateStageConfig()
    run_task(config)


if __name__ == "__main__":
    main()
