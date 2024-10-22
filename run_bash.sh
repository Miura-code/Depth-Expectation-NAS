#!/bin/bash
teacher_path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/noDepthLoss/s0-BaselineBestCell/best.pth.tar

# bash run_finetune.sh train FINETUNE2 efficientnet_v2_s pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0
# bash run_finetune.sh train FINETUNE2 efficientnet_v2_m pretrained pretrained_LR_features-0.001_classifier-0.01_cosine_warmup-0

experiment_name=SEARCHEVALnoDL
for seed in 0 1 2 3 4;do
    bash run_search3.sh SearchEval train stage ${experiment_name} none none s${seed}-baselinebest BASELINE_BEST Search_and_Evaluate_continuesly_noDepthLoss_nonKD_sw3 ${seed} 0 0
done

# minl=0.004
# for l in 0.00001 0.0001;do
#     bash run_search3.sh ArchKD train stage ${experiment_name} h_das_224baseline $teacher_path l${l}-${minl}-h_das_teacher BASELINE_BEST ArchHint_KD_mimic_only_teacher_archtecture ${seed} $l ${minl}
# done

# for l in 0.01;do
#     bash run_search3.sh ArchKD train stage ${experiment_name} h_das_224baseline $teacher_path l${l}-2-h_das_teacher BASELINE_BEST ArchHint_KD_mimic_only_teacher_archtecture 0 ${l}
# done

# bash run_search.sh train stage ORIGINAL none none s0-baselinebest BASELINE_BEST noKD_original_H-DAS_model_baselinebest-cell 0

# dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/ORIGINAL/s0-baselinebest/DAG
# dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# for seed in 0 1 2 3 4;do
#     bash run_evaluate.sh train stage ORIGINAL none none BASELINE_BEST $dag s${seed}-eval-baselinebest noKD_original_H-DAS_model_baselinebest-cell-stage $seed
#     bash run_evaluate.sh test stage BASELINE_BEST ${dag} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/ORIGINAL/s${seed}-eval-baselinebest/best.pth.tar
# done

# dagdirs=(
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/ARCH-KD/l0.01-2-h_das_teacher/DAG/EP50-best.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/ARCH-KD/l0.001-2-h_das_teacher/DAG/EP45-best.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/ARCH-KD/l0.0001-2-h_das_teacher/DAG/EP49-best.pickle
# )
# resumedirs=(
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/ARCH-KD/l0.01-2-h_das_teacher/best.pth.tar
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/ARCH-KD/l0.001-2-h_das_teacher/best.pth.tar
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/ARCH-KD/l0.0001-2-h_das_teacher/best.pth.tar
# )
# for seed in 0 1 2 3; do
#     for i in "${!dagdirs[@]}"; do
#         dag_path=${dagdirs[i]}
#         resume_path=${resumedirs[i]}
#         l=$(echo "$dag_path" | sed -n 's|.*ARCH-KD/l\([^/]*\)-2-h.*|\1|p')
#         bash run_evaluate2.sh train stage ${experiment_name} none none BASELINE_BEST ${dag_path} s${seed}l${l}-relax ArchHint_KD_mimic_only_teacher_archtecture ${seed} ${resume_path}
#         bash run_evaluate2.sh test stage BASELINE_BEST ${dag_path} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/${experiment_name}/s${seed}l${l}-relax/best.pth.tar
#     done
# done

# dirs=(
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/${experiment_name}/s0-freezeOtherLayer/DAG
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/${experiment_name}/s0-3stageFreezeOther/DAG
# )
# for dir in ${dirs[@]}; do
#     # ディレクトリ内で「best」を含むファイルをリストアップし、最も新しいファイルを見つける
#     dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
#     echo $newest_file
#     seed=$(echo "$dag" | sed -n 's|.*'${experiment_name}'/s\([^/]*\)-Baseline.*|\1|p')
#     exp_name=$(echo "$dag" | sed -n 's|.*s0-\([^/]*\)/DAG.*|\1|p')
#     bash run_evaluate.sh train stage ${experiment_name} non non BASELINE_BEST ${dag} s$seed-${exp_name} search_architecture_on_HINT_KD_for_all_parameters_without_depth_loss_with_large_slidewindow 0
#     bash run_evaluate.sh test stage BASELINE_BEST ${dag} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/${experiment_name}/s$seed-${exp_name}/best.pth.tar
# done

# for seed in 1 2 3 4;do
#     bash run_search2.sh train stage ${experiment_name} efficientnet_v2_s $teacher_path s${seed}-BaselineBestCell BASELINE_BEST search_architecture_on_KD_for_architecture_parameters_without_depth_loss_with_large_slidewindow ${seed}
# done

# dirs=(
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/${experiment_name}/s1-BaselineBestCell/DAG
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/${experiment_name}/s2-BaselineBestCell/DAG
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/${experiment_name}/s3-BaselineBestCell/DAG
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/${experiment_name}/s4-BaselineBestCell/DAG
# )
# for dir in ${dirs[@]}; do
#     # ディレクトリ内で「best」を含むファイルをリストアップし、最も新しいファイルを見つける
#     dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
#     echo $newest_file
#     seed=$(echo "$dag" | sed -n 's|.*'${experiment_name}'/s\([^/]*\)-Baseline.*|\1|p')
#     bash run_evaluate.sh train stage ${experiment_name} non non BASELINE_BEST ${dag} s$seed-BaselineBestCell search_architecture_on_HINT_KD_for_all_parameters_without_depth_loss_with_large_slidewindow 0
#     bash run_evaluate.sh test stage BASELINE_BEST ${dag} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/${experiment_name}/s$seed-BaselineBestCell/best.pth.tar
# done

# Ls=(0.3 0.4 0.5 0.6 0.7)
# Ts=(10 20)
# for t in ${Ts[@]}; do
#     bash run_evaluate.sh train cell KD_VALID_NEW efficientnet_v2_s /home/miura/lab/KD-hdas/results/teacher/cifar100/efficientnet_v2_s/FINETUNE2/pretrained-20240716-002108/best.pth.tar BASELINE224 l0.3T${t} T^2_to_soft_loss 0.3 ${t}
# # done

# dags=(
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar10/noSWDL/s0-BaselineBestCell/DAG/EP47-best.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar10/noSWDL/s1-BaselineBestCell/DAG/EP48-best.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar10/noSWDL/s2-BaselineBestCell/DAG/EP49-best.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar10/noSWDL/s3-BaselineBestCell/DAG/EP44-best.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar10/noSWDL/s4-BaselineBestCell/DAG/EP50-best.pickle
# )
# for dag in ${dags[@]}; do
#     extracted1=$(echo "$dag" | sed -n 's|.*-\([^/]*\)/DAG.*|\1|p')
#     extracted2=$(echo "$dag" | sed -n 's|.*SWDL/\([^/]*\)-Baseline.*|\1|p')
#     echo $extracted1$extracted2
#     bash run_evaluate.sh train stage noSlideWindow non non BASELINE_BEST ${dag} ${extracted1}s$extracted2 stage_architecure_evaluation_on_nonKD_searching_without_slidingWindow 0
# done

# bash run_evaluate.sh train stage BASELINE_TEST non non BASELINE_BEST NonDepth_BASELINE nonDepthEval non_depth_loss_stage_architecture 0
# bash run_evaluate.sh train stage BASELINE_TEST non non BASELINE_BEST NonVArch_BASELINE nonVArchEval non_virtual_architecture_step_on_stage_architecture 0
# bash run_evaluate.sh train stage BASELINE_TEST non non BASELINE_BEST BASELINE_BEST_STAGE BEST search_stage_on_baseline_best_cell 0

# for seed in 1 2 3 4;do
#     bash run_evaluate.sh train stage BASELINE_TEST non non BASELINE_BEST NonDepth_BASELINE nonDepthEvals${seed} non_depth_loss_stage_architecture $seed
#     bash run_evaluate.sh train stage BASELINE_TEST non non BASELINE_BEST NonVArch_BASELINE nonVArchEvals${seed} non_virtual_architecture_step_on_stage_architecture ${seed}
#     bash run_evaluate.sh train stage BASELINE_TEST non non BASELINE_BEST BASELINE_BEST_STAGE BASELINE_BEST_STAGE${seed} search_stage_on_baseline_best_cell ${seed}

# done
