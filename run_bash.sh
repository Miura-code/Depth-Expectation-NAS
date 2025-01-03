# #!/bin/bash
# teacher_path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/noDepthLoss/s0-BaselineBestCell/best.pth.tar
genotype=BASELINE_BEST

experiment_name=Pruning

## 構造のMACを計算する

search_dir="/home/miura/lab/KD-hdas/results"
regex="search_stage_KD/cifar100/Curriculum/s[0-9]-expected2-sw3-g-1_30-30/test/test-discrete-[0-9]{8}-[0-9]{6}/dag\.pickle"

# 結果を格納する配列
declare -a dags

# ファイルを検索して正規表現に一致するものを配列に格納
while IFS= read -r file; do
    dags+=("$file")
done < <(find "$search_dir" -type f | grep -E "$regex")

# 配列の内容を表示（確認用）
echo "Found files:"
for file in "${dags[@]}"; do
    echo "$file"
done

# dags=(
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/Curriculum/s0-expected2-sw3-g-0.01_30-30/test/test-discrete-20250103-093946/dag.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/Curriculum/s1-expected2-sw3-g-0.01_30-30/test/test-discrete-20250103-174101/dag.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/Curriculum/s2-expected2-sw3-g-0.01_30-30/test/test-discrete-20250104-014927/dag.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/Curriculum/s3-expected2-sw3-g-0.01_30-30/test/test-discrete-20250104-015015/dag.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/Curriculum/s4-expected2-sw3-g-0.01_30-30/test/test-discrete-20250103-180510/dag.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/Curriculum/s0-expected2-sw3-g-0.001_30-30/test/test-discrete-20250101-173910/dag.pickle
#     /home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/Curriculum/s1-expected2-sw3-g-0.001_30-30/test/test-discrete-20250102-012221/dag.pickle

# )   

for dag in "${dags[@]}" ; do
    genotype=$1
    DAG=$dag
    channels=16
    layers=32
    python cal_macs_main.py \
        --genotype $genotype\
        --DAG $DAG\
        --spec_cell \
        --init_channels $channels\
        --layers $layers\
        --aux_weight 0.0\
        --drop_path_prob 0.0\
        --dataset cifar100\
        --advanced
    done


    

# for seed in 0;do
#     for g in 0.001 0.01 0.1 1;do
#     # for g in 0.001;do
#         for method in expected;do
#             # bash run_searchStage.sh train Distribution \
#             # $experiment_name none none s$seed-depth-sw3-g$g \
#             # $genotype \
#             # search_and_evaluation_curriculum_learning_with_-Beta-expected-Depth-ecpected-constraint_slidewindow-3 \
#             # $seed 0 0 0\
#             # 1 $g 3 1 0\
#             # $method "30 30"

#             bash run_searchStage.sh train Distribution \
#             ${experiment_name} none none s$seed-depth-sw3-g$g\
#             $genotype\
#             search_and_evaluation_curriculum_learning_with_-Beta-expected-Depth-ecpected-constraint_slidewindow-3\
#             $seed 0 0 0\
#             1 ${g} 3 1 0\
#             $method "30 30"

#             dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-depth-sw3-g$g/DAG
#             dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
#             path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-depth-sw3-g$g/best.pth.tar
#             bash run_searchStage.sh test Distribution \
#             $path BASELINE_BEST $dag \
#             3 1

# #             # dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g$g/DAG
# #             # dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# #             # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g$g/best.pth.tar

# #             # bash run_evaluate.sh train stage Pruning \
# #             # none none \
# #             # $genotype $dag \
# #             # s$seed-$method-sw3-g$g \
# #             # search_with_L1-Alpha-constraint_slidewindow-3 \
# #             # 0

# #             # path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g$g/best.pth.tar
# #             # bash run_evaluate.sh test stage $genotype $dag $path
#         done
# #         # bash run_searchStage.sh train ArchKD \
# #         # $experiment_name h_das_224baseline $teacher_path s$seed-alphal1-sw3-g$g\
# #         # $genotype \
# #         # search_with_L1-Alpha-constraint_slidewindow-3 \
# #         # $seed $g $g $g\
# #         # 1 0 3 0 0\
# #         # alphal1

# #         # dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-alphal1-sw3-g$g/DAG
# #         # dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# #         # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-alphal1-sw3-g$g/best.pth.tar

# #         # bash run_evaluate.sh train stage ArchKD \
# #         # none none \
# #         # $genotype $dag \
# #         # s$seed-alphal1-sw3-g$g \
# #         # search_with_L1-Alpha-constraint_slidewindow-3 \
# #         # 0

# #         # path=/home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/$experiment_name/s$seed-alphal1-sw3-g$g/best.pth.tar
# #         # bash run_evaluate.sh test stage $genotype $dag $path
#     done
# done

# # dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s0-BaselineBestCell/DAG
# # dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# # bash run_evaluate.sh test stage BASELINE_BEST ${dag} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/noDepthLoss/s0-noAux16ch/best.pth.tar
# # for seed in 0 1 2;do
# #     dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-discreteEval/DAG
# #     dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# #     # path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s${seed}-BaselineBestCell/best.pth.tar
# #     bash run_evaluate.sh train stage $experiment_name none none BASELINE_BEST $dag s${seed}-discrete-noAux16ch-reset evaluate_without_auxiliary_head_and_init_channel-16_droppathprob-0 $seed
# #     bash run_evaluate.sh test stage BASELINE_BEST ${dag} /home/miura/lab/KD-hdas/results/evaluate_stage_KD/cifar100/$experiment_name/s${seed}-discrete-noAux16ch-reset/best.pth.tar
# # done

# # method=expected
# # for seed in 0 1 2;do
# #     bash run_searchStage.sh train SearchEvalCurriculum \
# #     $experiment_name none none s$seed-$method-sw3-g${g}_0-50 \
# #     BASELINE_BEST \
# #     search_and_evaluation_curriculum_learning_with_-Beta-expected-Depth-ecpected-constraint_slidewindow-3_ \
# #     $seed 0 0 0.001\
# #     1 0 3 1 0 \
# #     expected "0 50"

# #     dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g${g}_0-50/DAG
# #     dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# #     path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-$method-sw3-g${g}_0-50/best.pth.tar
# #     bash run_searchStage.sh test SearchEvalCurriculum \
# #     $path BASELINE_BEST $dag \
# #     3 1
# # done

# # experiment_name=SEARCHEVALnoDL

# # method=expected
# # for seed in 0 1 2 3 4 5;do
# #     bash run_searchStage.sh train SearchEval \
# #     $experiment_name none none s$seed-L32 \
# #     BASELINE_BEST \
# #     search_and_evaluation_curriculum_learning_with_-Beta-expected-Depth-ecpected-constraint_slidewindow-3_ \
# #     $seed 0 0 0.001\
# #     1 0 3 1 0 \
# #     $method "0 50"

# #     dir=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-L32/DAG
# #     dag=$(find "$dir" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
# #     path=/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/$experiment_name/s$seed-L32/best.pth.tar
# #     bash run_searchStage.sh test SearchEval \
# #     $path BASELINE_BEST $dag \
# #     3 1
# # done