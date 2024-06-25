
'''
Fine tune teacher model for knowledge distillation using timm
'''
import os
from config.trainTeacher_config import TrainTeacherConfig
from trainer.finetuneTeacher_trainer import TrainTeacherTrainer
import utils
from utils.eval_util import RecordDataclass
from utils.logging_util import get_std_logging

LOSS_TYPES = ["training_loss", "validation_loss"]
ACC_TYPES = ["training_accuracy", "validation_accuracy"]

def run_task(config):
    logger = get_std_logging(os.path.join(config.path, "{}.log".format(config.name)))
    config.logger = logger

    config.print_params(logger.info)
    
    # set seed
    utils.set_seed_gpu(config.seed, config.gpus[0])

    trainer = TrainTeacherTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch
    
    # loss, accを格納する配列
    Record = RecordDataclass(LOSS_TYPES, ACC_TYPES)

    best_top1 = 0.
    for epoch in range(start_epoch, trainer.total_epochs):
        train_top1, train_loss = trainer.train_epoch(epoch, printer=logger.info)
        val_top1, val_loss = trainer.val_epoch(epoch, printer=logger.info)
        trainer.lr_scheduler.step()

        trainer.writer.add_scalar('train/lr', round(trainer.lr_scheduler.get_last_lr()[0], 5), epoch)
        trainer.writer.add_scalar('train/loss', train_loss, epoch)
        trainer.writer.add_scalar('train/top1', train_top1, epoch)
        # trainer.writer.add_scalar('train/top5', prec5.item(), epoch)

        trainer.writer.add_scalar('val/loss', val_loss, epoch)
        trainer.writer.add_scalar('val/top1', val_top1, epoch)
        # trainer.writer.add_scalar('val/top5', top5.avg, epoch)

    
        if best_top1 < val_top1:
            best_top1, is_best = val_top1, True
        else:
            is_best = False
        trainer.save_checkpoint(epoch, is_best=is_best)
        Record.add(LOSS_TYPES+ACC_TYPES, [train_loss, val_loss, train_top1, val_top1])
        Record.save(config.path)
        logger.info("Until now, best Prec@1 = {:.4%}".format(best_top1))
    
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    

    

def main():
    config = TrainTeacherConfig()
    run_task(config)


if __name__ == "__main__":
    main()
