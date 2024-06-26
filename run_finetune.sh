#!/bin/bash

name=$1

# ===== セルレベル探索　=====
teacher_model=$2
dataset=cifar10
cutout=0
batch_size=64
epoch=25
seed=0
train_portion=1.0
# python finetuneTeacher_main.py \
#     --name $name \
#     --model_name $teacher_model\
#     --dataset $dataset\
#     --cutout_length $cutout\
#     --batch_size $batch_size \
#     --epochs $epoch \
#     --seed $seed \
#     --save TEST \
#     --train_portion $train_portion

# for epoch in 100 200 300; do
#     python finetuneTeacher_main.py \
#         --name $name \
#         --model_name $teacher_model\
#         --dataset $dataset\
#         --cutout_length $cutout\
#         --batch_size $batch_size \
#         --epochs $epoch \
#         --seed $seed \
#         --save E$epoch \
#         --train_portion $train_portion
# done

# teachers=("densenet121" "densenet161" "resnet50" "resnet152")
teachers=("wide_resnet50_2" "tf_efficientnetv2_s" "tf_efficientnetv2_m")

for teacher_model in "${teachers[@]}"; do
    python finetuneTeacher_main.py \
        --name $name \
        --model_name $teacher_model\
        --dataset $dataset\
        --cutout_length $cutout\
        --batch_size $batch_size \
        --epochs $epoch \
        --seed $seed \
        --save E$epoch \
        --train_portion $train_portion
done