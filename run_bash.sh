#!/bin/bash
teacher_path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/noDepthLoss/s0-BaselineBestCell/best.pth.tar
genotype=BASELINE_BEST

experiment_name=Pruning

for seed in 0 1 2 3 4;do
    for g in 15 20;do
        for method in l1 length;do
            bash run_searchStage.sh train Pruning \
            $experiment_name none none s$seed-$method-sw3-g$g\
            $genotype \
            search_with_L1-Alpha-constraint_slidewindow-3 \
            $seed 0 0 $g\
            1 0 3 0 0\
            $method

            # dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g$g/DAG
            # dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
            # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g$g/best.pth.tar

            # bash run_evaluate.sh train stage Pruning \
            # none none \
            # $genotype $dag \
            # s$seed-$method-sw3-g$g \
            # search_with_L1-Alpha-constraint_slidewindow-3 \
            # 0

            # path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g$g/best.pth.tar
            # bash run_evaluate.sh test stage $genotype $dag $path
        done
        bash run_searchStage.sh train ArchKD \
        $experiment_name h_das_224baseline $teacher_path s$seed-alphal1-sw3-g$g\
        $genotype \
        search_with_L1-Alpha-constraint_slidewindow-3 \
        $seed $g $g $g\
        1 0 3 0 0\
        alphal1

        # dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-alphal1-sw3-g$g/DAG
        # dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
        # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-alphal1-sw3-g$g/best.pth.tar

        # bash run_evaluate.sh train stage ArchKD \
        # none none \
        # $genotype $dag \
        # s$seed-alphal1-sw3-g$g \
        # search_with_L1-Alpha-constraint_slidewindow-3 \
        # 0

        # path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/$experiment_name/s$seed-alphal1-sw3-g$g/best.pth.tar
        # bash run_evaluate.sh test stage $genotype $dag $path
    done
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
