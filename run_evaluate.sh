#!/bin/bash

name=$1

# ===== セルレベルアーキテクチャを評価　=====
# teacher_model=$2
# teacher_path=$3
# genotype=$4
# save=$5
# dataset=cifar100
# lambda=0.6
# T=3
# batch_size=64
# epoch=100
# train_portion=0.9
# seed=0
# python evaluateCell_main.py \
#     --name $name \
#     --genotype $genotype \
#     --teacher_name $teacher_model\
#     --teacher_path $teacher_path \
#     --l $lambda\
#     --T $T \
#     --dataset $dataset\
#     --batch_size $batch_size \
#     --epochs $epoch \
#     --train_portion $train_portion \
#     --seed $seed \
#     --save $save \
#     --nonkd

# ## セルレベル構造のテスト
genotype=$1
path=$2
seed=0

python testCell_main.py \
    --save test \
    --dataset cifar100 \
    --batch_size 128 \
    --genotype $genotype \
    --seed $seed \
    --resume_path $path 