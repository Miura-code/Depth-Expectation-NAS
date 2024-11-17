#!/bin/bash
teacher_path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/noDepthLoss/s0-BaselineBestCell/best.pth.tar
genotype=BASELINE_BEST

experiment_name=Curriculum

g=10

for seed in 0;do
    bash run_searchStage.sh train SearchEvalCurriculum \
    $experiment_name none none s$seed-g$g-30-20\
    $genotype \
    search_eval_beta_concat_currisulum_betal1-criterion_ \
    $seed 0 0 $g\
    1 0 12 1 0\
    length "30 20"

    bash run_searchStage.sh train SearchEvalCurriculum \
    $experiment_name none none s$seed-g$g-50-0\
    $genotype \
    search_eval_beta_concat_currisulum_betal1-criterion_ \
    $seed 0 0 $g\
    1 0 12 1 0\
    length "50 0"

    bash run_searchStage.sh train SearchEvalCurriculum \
    $experiment_name none none s$seed-g$g-0-50\
    $genotype \
    search_eval_beta_concat_currisulum_betal1-criterion_ \
    $seed 0 0 $g\
    1 0 12 1 0\
    length "0 50"
    
    # dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-sw3-g0/DAG
    # dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
    # # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-L29/best.pth.tar    
    
    # bash run_evaluate.sh train stage $experiment_name none none \
    # $genotype $dag \
    # s$seed-sw3-g0 \
    # evaluate_search_stage_32-layer-network \
    # 0

    # bash run_evaluate.sh test stage \
    # $genotype $dag \
    # /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/$experiment_name/s$seed-sw3-g0/best.pth.tar
    
    # bash run_searchStage.sh test KD \
    # $path $genotype $dag \
    # 3 1
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
