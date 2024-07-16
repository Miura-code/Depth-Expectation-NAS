#!/bin/bash

type=$1
if [ ${type} = "train" ]; then
    # ===== セルレベルアーキテクチャを評価　=====
    name=$2
    teacher_model=$3
    teacher_path=$4
    genotype=$5
    save=$5
    dataset=cifar100
    lambda=0.6
    T=3
    batch_size=64
    epoch=100
    train_portion=0.9
    seed=0
    python evaluateCell_main.py \
        --name $name \
        --genotype $genotype \
        --teacher_name $teacher_model\
        --teacher_path $teacher_path \
        --l $lambda\
        --T $T \
        --dataset $dataset\
        --batch_size $batch_size \
        --epochs $epoch \
        --train_portion $train_portion \
        --seed $seed \
        --save $save \
        # --nonkd
elif [ ${type} = "test" ]; then
    ## セルレベル構造のテスト
    genotype=$2
    path=$3
    seed=0

    python testCell_main.py \
        --save test \
        --dataset cifar100 \
        --batch_size 128 \
        --genotype $genotype \
        --seed $seed \
        --resume_path $path 
else
    echo ""
fi