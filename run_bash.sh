#!/bin/bash

# bash run_finetune.sh train FINETUNE2 efficientnet_v2_s pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0
# bash run_finetune.sh train FINETUNE2 efficientnet_v2_m pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0

# for seed in 3 ;do
#     bash run_search.sh train cell BASELINE non non BASELINE224 baseline_size224version_nonkd ${seed}
# done

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Ls=(0.3 0.4 0.5 0.6 0.7)
# Ts=(10 20)
# for t in ${Ts[@]}; do
#     bash run_evaluate.sh train cell KD_VALID_NEW efficientnet_v2_s /home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar BASELINE224 l0.3T${t} T^2_to_soft_loss 0.3 ${t}
# done
=======
Ls=(0.3 0.4 0.5 0.6 0.7)
Ts=(30)
>>>>>>> 0796289 (experiment: KD validation)

# teacher_path=/home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar

# for seed in 0 1 2 3 4;do
#     bash run_search.sh train cell ARCH_WEIGHT efficientnet_v2_s ${teacher_path} s${seed} search_cell_with_simple_kd_on_seed-${seed} ${seed}
# done

teacher_path=/home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar
genotypes=(
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240725-164733/DAG/EP47-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240731-193109/DAG/EP45-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-002228/DAG/EP48-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-093713/DAG/EP50-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-144014/DAG/EP46-best.pickle 
)
for genotype in ${genotypes[@]};do
    extracted=$(echo "$genotype" | sed -n 's|.*-\([^/]*\)/DAG.*|\1|p')
    bash run_evaluate.sh train cell ONLY_EVAL efficientnet_v2_s $teacher_path $genotype $extracted genotype-$genotype_KD_for_only_evaluation_stage 0.5 10 0
=======
# Ls=(0.3 0.4 0.5 0.6 0.7)
# Ts=(10 20)

# for t in ${Ts[@]}; do
#     bash run_evaluate.sh train cell KD_VALID_NEW efficientnet_v2_s /home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar BASELINE224 l0.3T${t} T^2_to_soft_loss 0.3 ${t}
# done

<<<<<<< HEAD
<<<<<<< HEAD
# genotypes=(/home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240731-193109/DAG/EP45-best.pickle /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-002228/DAG/EP48-best.pickle /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-093713/DAG/EP50-best.pickle /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-144014/DAG/EP46-best.pickle)
# for genotype in ${genotypes[@]}; do
#     save=$(echo "$genotype" | sed -n 's|.*-\([^/]*\)/DAG.*|\1|p')
#     # echo $save
#     bash run_evaluate.sh train cell BASELINE224 non non ${genotype} $save baseline224 3 3
# done

genotype=/home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-093713/DAG/EP50-best.pickle
for seed in 1 2 3 4;do
    bash run_evaluate.sh train cell BASELINE224 non non ${genotype} 093713s${seed} baseline224_genotype-BASELINE224-20240801-093713 0.3 3 ${seed}
>>>>>>> 7a96e72 (experiment: ベースラインの評価)
=======
teacher_path=/home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar
<<<<<<< HEAD
for seed in 0 1 2 3 4; do
    bash run_search.sh train cell ONLY_ARCH efficientnet_v2_s ${teacher_path} s${seed} search_cell_architecture_using_KD_for_architecture_parameter_optimization_with_seed-${seed} ${seed}
>>>>>>> 094865b (experiment: セルレベル探索にKDを導入)
=======

for seed in 0 1 2 3 4;do
    bash run_search.sh train cell ARCH_WEIGHT efficientnet_v2_s ${teacher_path} s${seed} search_cell_with_simple_kd_on_seed-${seed} ${seed}
>>>>>>> a6ddf01 (experiment: セルレベルの探索にKDを導入)
=======
# teacher_path=/home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar

# for seed in 0 1 2 3 4;do
#     bash run_search.sh train cell ARCH_WEIGHT efficientnet_v2_s ${teacher_path} s${seed} search_cell_with_simple_kd_on_seed-${seed} ${seed}
# done

teacher_path=/home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar
genotypes=(
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240725-164733/DAG/EP47-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240731-193109/DAG/EP45-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-002228/DAG/EP48-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-093713/DAG/EP50-best.pickle
    /home/miura/lab/KD-hdas/results/search_cell_KD/cifar100/BASELINE/BASELINE224-20240801-144014/DAG/EP46-best.pickle 
)
for genotype in ${genotypes[@]};do
    extracted=$(echo "$genotype" | sed -n 's|.*-\([^/]*\)/DAG.*|\1|p')
    bash run_evaluate.sh train cell ONLY_EVAL efficientnet_v2_s $teacher_path $genotype $extracted genotype-$genotype_KD_for_only_evaluation_stage 0.5 10 0
>>>>>>> 4fee45d (experiment: セルレベル探索の評価)
=======
for seed in 1 2 3 4;do
=======
for seed in 3 ;do
>>>>>>> 9e3ae3b (experiment: BASELINE224 cell-levelを追加)
    bash run_search.sh train cell BASELINE non non BASELINE224 baseline_size224version_nonkd ${seed}
<<<<<<< HEAD
>>>>>>> 8dcb21c (experiment; BASELINE224 search cell)
=======
done
=======
# for seed in 3 ;do
#     bash run_search.sh train cell BASELINE non non BASELINE224 baseline_size224version_nonkd ${seed}
# done
>>>>>>> 0796289 (experiment: KD validation)

Ls=(0.3 0.4 0.5 0.6 0.7)
Ts=(30)

for t in ${Ts[@]}; do
    bash run_evaluate.sh train cell KD_VALID_NEW efficientnet_v2_s /home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar BASELINE224 l0.3T${t} T^2_to_soft_loss 0.3 ${t}
>>>>>>> 33c7fbc (experiment: KD_VALIDATION)
=======
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
>>>>>>> 7a96e72 (experiment: ベースラインの評価)
done