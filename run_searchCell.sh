#!/bin/bash

name=$1

# ===== セルレベル探索　=====
teacher_model=$2
teacher_path=$3
save=$4
dataset=cifar100
lambda=0.6
T=3
batch_size=64
epoch=50
train_portion=0.5 # searchの場合train_portionは0.5が最大値
seed=0
python searchCell_KD_main.py \
    --name $name \
    --teacher_name $teacher_model\
    --teacher_path $teacher_path \
    --l $lambda\
    --T $T \
    --dataset $dataset\
    --batch_size $batch_size \
    --epochs $epoch \
    --train_portion $train_portion \
    --seed $seed \
    --save $save


## ===== モデルをテスト =====
# save=$1
# resume_path=$2
# genotype=$3
# dataset=cifar100
# cutout=0
# batch_size=64
# seed=0
# train_portion=1.0
# # teachers=("densenet121" "densenet161" "resnet50" "resnet152")
# teachers=("wide_resnet50_2" "tf_efficientnetv2_s" "tf_efficientnetv2_m")
# python testCell_main.py \
#         --save $save \
#         --resume_path $resume_path \
#         --genotype $genotype \
#         --dataset $dataset\
#         --seed $seed