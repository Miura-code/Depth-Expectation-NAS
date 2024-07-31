#!/bin/bash

# bash run_finetune.sh train FINETUNE2 efficientnet_v2_s pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0
# bash run_finetune.sh train FINETUNE2 efficientnet_v2_m pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0

for seed in 3 ;do
    bash run_search.sh train cell BASELINE non non BASELINE224 baseline_size224version_nonkd ${seed}
done

Ls=(0.3 0.4 0.5 0.6 0.7)
Ts=(3 4 5 6 7)

for t in ${Ts[@]}; do
    bash run_evaluate.sh train cell KD_VALID_NEW efficientnet_v2_s /home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar BASELINE224 l0.3T${t} T^2_to_soft_loss 0.3 ${t}
done