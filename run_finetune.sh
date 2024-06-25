#!/bin/bash

name=$1

# ===== セルレベル探索　=====
teacher_model=$2
dataset=cifar10
cutout=0
batch_size=64
epoch=2
seed=0
train_portion=0.1
python finetuneTeacher_main.py \
    --name $name \
    --model_name $teacher_model\
    --dataset $dataset\
    --cutout_length $cutout\
    --batch_size $batch_size \
    --epochs $epoch \
    --seed $seed \
    --save TEST \
    --train_portion $train_portion

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