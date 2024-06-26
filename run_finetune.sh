#!/bin/bash

# ===== モデルをファインチューニング =====
# name=$1

# teacher_model=$2
# dataset=cifar10
# cutout=0
# batch_size=64
# epoch=25
# seed=0
# train_portion=1.0
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

## ===== 複数エポック数でファインチューニングする =====
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

## ===== 複数モデルをファインチューニングする =====
# teachers=("densenet121" "densenet161" "resnet50" "resnet152")
# teachers=("wide_resnet50_2" "tf_efficientnetv2_s" "tf_efficientnetv2_m")

# for teacher_model in "${teachers[@]}"; do
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

## ===== モデルをテスト =====
name=$1
dataset=cifar10
seed=0
teachers=("densenet121" "densenet161" "resnet50" "resnet152" "wide_resnet50_2" "tf_efficientnetv2_s" "tf_efficientnetv2_m")
paths=("results/teacher/cifar10/densenet121/FINETUNE/E25-20240626-151440/best.pth.tar" \
results/teacher/cifar10/densenet161/FINETUNE/E25-20240626-153719/best.pth.tar \
results/teacher/cifar10/resnet50/FINETUNE/E25-20240626-160747/best.pth.tar \
results/teacher/cifar10/resnet152/FINETUNE/E25-20240626-162157/best.pth.tar \
results/teacher/cifar10/wide_resnet50_2/FINETUNE/E25-20240626-151514/best.pth.tar \
results/teacher/cifar10/tf_efficientnetv2_s/FINETUNE/E25-20240626-152903/best.pth.tar \
results/teacher/cifar10/tf_efficientnetv2_m/FINETUNE/E25-20240626-155425/best.pth.tar
)

for i in $(seq 0 $((${#teachers[@]} - 1)));do
    teacher_model=${teachers[$i]}
    resume_path=${paths[$i]}
    python testTeacher_main.py \
        --name $name \
        --model_name $teacher_model \
        --resume_path $resume_path \
        --dataset $dataset\
        --seed $seed
done