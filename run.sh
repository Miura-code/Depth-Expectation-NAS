#!/bin/bash

name=$1

stage_architecture=("HS_DAS_CIFAR" "HS_DAS_CIFAR_SKIP" "STAGE_HSDAS_V1" "STAGE_HSDAS_V2" "STAGE_HSDAS_V3" "STAGE_SHALLOW" "STAGE_MIDDLE" "STAGE_DEEP" "STAGE_DARTS" )
stage_architecture=("HS_DAS_CIFAR" "HS_DAS_CIFAR_SKIP" "STAGE_SHALLOW" "STAGE_MIDDLE" "STAGE_DEEP" "STAGE_DARTS" "STAGE_FULL_CASCADE")

batch_size=64
epoch=50
train_portion=0.5 # searchの場合train_portionは0.5が最大値
seed=0

# ステージの探索
# python searchStage_main.py \
#     --name $name \
#     --batch_size $batch_size \
#     --dataset CIFAR10 \
#     --epochs $epoch \
#     --genotype HC_DAS \
#     --train_portion $train_portion \
#     --seed $seed \
#     --spec_cell

for seed in 0 1 2 3; do
    echo $arch
    python searchStage_main.py \
        --name $name \
        --batch_size $batch_size \
        --dataset CIFAR10 \
        --epochs $epoch \
        --genotype PDARTS \
        --train_portion $train_portion \
        --seed $seed \
        --spec_cell
done

## 事前学習アーキテクチャからステージの探索
# checkpoint_path=$2
# python searchStage_main.py \
#     --name $name \
#     --batch_size $batch_size \
#     --dataset CIFAR10 \
#     --epochs $epoch \
#     --genotype DARTS_V1 \
#     --train_portion $train_portion \
#     --seed $seed

# for seed in 0 1 2 3; do
#     echo $arch
#     python searchStage_main.py \
#         --name $name \
#         --batch_size $batch_size \
#         --dataset CIFAR10 \
#         --epochs $epoch \
#         --genotype DARTS_V1 \
#         --train_portion $train_portion \
#         --seed $seed \
# done

## 事前学習アーキテクチャからステージの探索
# checkpoint_path=$2
# python searchStage_main.py \
#     --name $name \
#     --batch_size $batch_size \
#     --dataset CIFAR10 \
#     --epochs $epoch \
#     --genotype DARTS_V1 \
#     --train_portion $train_portion \
>>>>>>> ex/miura
#     --seed $seed \
#     --share_stage \
#     --resume_path $checkpoint_path \
#     --checkpoint_reset

# checkpoints=(/home/miura/lab/research-hdas/results/search_Stage/mnist/SCRATCH/EXP-20240525-152620/best.pth.tar /home/miura/lab/research-hdas/results/search_Stage/mnist/SCRATCH/EXP-20240525-175220/best.pth.tar /home/miura/lab/research-hdas/results/search_Stage/mnist/SCRATCH/EXP-20240525-201945/best.pth.tar /home/miura/lab/research-hdas/results/search_Stage/mnist/SCRATCH/EXP-20240525-224521/best.pth.tar)
# for checkpoint_path in "${checkpoints[@]}"; do
#     echo $checkpoint_path
#     exp_id=$(echo "$checkpoint_path" | grep -oP '\d{8}-\d{6}')  
#     echo "EXP($exp_id)"
#     python searchStage_main.py \
#         --name $name \
#         --batch_size $batch_size \
#         --dataset CIFAR10 \
#         --epochs $epoch \
#         --genotype DARTS_V1 \
#         --train_portion $train_portion \
#         --seed $seed \
#         --share_stage \
#         --resume_path $checkpoint_path \
#         --checkpoint_reset \
#         --save "EXP($exp_id)"
# done

## ステージの評価・ファインチューニング
# python augmentStage_main.py \
#     --name $name  \
#     --batch_size $batch_size \
#     --dataset cifar10 \
#     --epochs $epoch \
#     --genotype DARTS_V1 \
#     --DAG HS_DAS_CIFAR \
#     --train_portion $train_portion \
#     --seed $seed \

# for seed in 1 2; do
#     for arch in "${stage_architecture[@]}"; do
#         echo $arch
#         python augmentStage_main.py \
#         --name $arch  \
#         --batch_size $batch_size \
#         --dataset cifar10 \
#         --epochs $epoch \
#         --genotype DARTS_V1 \
#         --DAG $arch \
#         --train_portion $train_portion \
#         --seed $seed \
#         --save E600
#     done
# done

# for arch in "${stage_architecture[@]}"; do
#     echo $arch
#     python augmentStage_main.py \
#     --name $arch  \
#     --batch_size $batch_size \
#     --dataset cifar10 \
#     --epochs $epoch \
#     --genotype DARTS_V1 \
#     --DAG $arch \
#     --train_portion $train_portion \
#     --seed $seed
# done


# python searchStage_main.py \
#     --name macro-cifar10-test \
#     --w_weight_decay 0.0027  \
#     --dataset cifar10 \
#     --batch_size 64 \
#     --workers 0  \
#     --genotype DARTS_V1

# ## ステージのテスト
# arch=$1
# seed=0

# path=$2
# python testStage_main.py \
#     --name test \
#     --dataset cifar10 \
#     --batch_size 128 \
#     --genotype DARTS_V1 \
#     --DAG $arch \
#     --seed $seed \
#     --resume_path $path

# ディレクトリパスを指定
# arch=$1
# target_directory=/home/miura/lab/research-hdas/results/search_Stage/CIFAR10/SCRATCH
# specific_name="EXP-"
# seed=0
# layer=20

# # # # ディレクトリ内の全てのディレクトリ名を取得し、test.pyを実行する
# for directory in "$target_directory"/*; do
#     # ディレクトリ名を変数に格納
#     directory_path="$directory"
#     if [[ "$directory_path" == *"$specific_name"* ]]; then
#         # ディレクトリ名を表示
#         echo "Directory name: $directory_path"
#         resume_path=$directory_path/best.pth.tar
#         echo $resume_path
    
#         python testStage_main.py \
#             --name test \
#             --dataset cifar10 \
#             --batch_size 128 \
#             --genotype DARTS_V1 \
#             --DAG $arch \
#             --seed $seed \
#             --resume_path $resume_path \
#             --layer $layer
#     fi
# done
#         python testStage_main.py \
#             --name test \
#             --dataset cifar10 \
#             --batch_size 128 \
#             --genotype DARTS_V1 \
#             --DAG $arch \
#             --seed $seed \
#             --resume_path $resume_path \
#             --layer $layer
#     fi
# done
# done
