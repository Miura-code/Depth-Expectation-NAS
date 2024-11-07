#!/bin/bash
teacher_path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/noDepthLoss/s0-BaselineBestCell/best.pth.tar
genotype=BASELINE_BEST

experiment_name=Pruning

for seed in 0;do
    bash run_searchStage.sh train ArchKD \
    $experiment_name h_das_224baseline $teacher_path s$seed-L1Alpha\
    $genotype \
    search_wtih_cell-length-constriction \
    $seed 0 1 1 \
    1 0 10 0 0\
    alphal1
    # bash run_searchStage.sh train Pruning \
    # $experiment_name none none s$seed-L1Beta\
    # $genotype \
    # search_wtih_beta-L2-constriction \
    # $seed 0 1 1 \
    # 1 0 10 0 0\
    # l2
    # dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-length-test/DAG
    # dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
    # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-length-test/best.pth.tar
    # bash run_searchStage.sh test Pruning \
    # $path $genotype $dag \
    # 10 0
done

# dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s0-BaselineBestCell/DAG
# dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# bash run_evaluate.sh test stage BASELINE_BEST ${dag} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/noDepthLoss/s0-noAux16ch/best.pth.tar
# for seed in 0 1 2;do
#     dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-discreteEval/DAG
#     dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
#     # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s${seed}-BaselineBestCell/best.pth.tar
#     bash run_evaluate.sh train stage $experiment_name none none BASELINE_BEST $dag s${seed}-discrete-noAux16ch-reset evaluate_without_auxiliary_head_and_init_channel-16_droppathprob-0 $seed
#     bash run_evaluate.sh test stage BASELINE_BEST ${dag} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/$experiment_name/s${seed}-discrete-noAux16ch-reset/best.pth.tar
# done

# for seed in 0 1 2 3 4;do
#     bash run_search3.sh SearchEval train stage $experiment_name none none s$seed-discreteEval-reset BASELINE_BEST search_and_evaluate_on_SearchingModel_discretized_arch_parameters_reset_ $seed 0.1 0
#     dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-discreteEval-reset/DAG
#     dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
#     path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s${seed}-discreteEval-reset/best.pth.tar
#     python testSearchedModel_main.py \
#         --resume_path $path \
#         --genotype BASELINE_BEST \
#         --DAG $dag \
#         --save test \
#         --seed 0 \
#         --spec_cell \
#         --slide_window 3 \
#         --discrete \
#         --advanced
# done
