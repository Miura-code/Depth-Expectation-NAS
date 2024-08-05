#!/bin/bash

# bash run_finetune.sh train FINETUNE2 efficientnet_v2_s pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0
# bash run_finetune.sh train FINETUNE2 efficientnet_v2_m pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0

# for seed in 3 ;do
#     bash run_search.sh train cell BASELINE non non BASELINE224 baseline_size224version_nonkd ${seed}
# done

# Ls=(0.3 0.4 0.5 0.6 0.7)
# Ts=(10 20)

# for t in ${Ts[@]}; do
#     bash run_evaluate.sh train cell KD_VALID_NEW efficientnet_v2_s /home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar BASELINE224 l0.3T${t} T^2_to_soft_loss 0.3 ${t}
# done

# genotypes=(/home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240731-193109/DAG/EP45-best.pickle /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-002228/DAG/EP48-best.pickle /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-093713/DAG/EP50-best.pickle /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-144014/DAG/EP46-best.pickle)
# for genotype in ${genotypes[@]}; do
#     save=$(echo "$genotype" | sed -n 's|.*-\([^/]*\)/DAG.*|\1|p')
#     # echo $save
#     bash run_evaluate.sh train cell BASELINE224 non non ${genotype} $save baseline224 3 3
# done

genotype=/home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-093713/DAG/EP50-best.pickle
for seed in 1 2 3 4;do
    bash run_evaluate.sh train cell BASELINE224 non non ${genotype} 093713s${seed} baseline224_genotype-BASELINE224-20240801-093713 0.3 3 ${seed}
done