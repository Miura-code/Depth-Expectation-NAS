#!/bin/bash

# ===== モデルをファインチューニング =====
# name=$1

# teacher_model=$2
# dataset=cifar10
# cutout=0
# batch_size=64
# epoch=2
# seed=0
# train_portion=0.1
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
# name=$1
# teacher_model=$2
# dataset=cifar10
# cutout=0
# batch_size=64
# epoch=50
# seed=0
# train_portion=1.0
# # teachers=("densenet121" "densenet161" "resnet50" "resnet152")
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
save=$1
teacher_model=$2
resume_path=$3
dataset=cifar10
seed=0
python testTeacher_main.py \
        --save $save \
        --model_name $teacher_model \
        --resume_path $resume_path \
        --dataset $dataset\
        --seed $seed

## ===== 複数モデルをテスト =====
# name=$1
# dataset=cifar100
# seed=0
# teachers=("densenet121" "densenet161" "resnet50" "resnet152" "wide_resnet50_2" "tf_efficientnetv2_s" "tf_efficientnetv2_m")
# paths=(
#     results/teacher/cifar100/densenet121/FINETUNE/E50-20240625-235942/best.pth.tar\
#     results/teacher/cifar100/densenet161/FINETUNE/E50-20240626-004503/best.pth.tar\
#     results/teacher/cifar100/resnet50/FINETUNE/E50-20240626-014606/best.pth.tar\
#     results/teacher/cifar100/resnet152/FINETUNE/E50-20240626-021142/best.pth.tar\
#     results/teacher/cifar100/wide_resnet50_2/FINETUNE/E50-20240626-001557/best.pth.tar\
#     results/teacher/cifar100/tf_efficientnetv2_s/FINETUNE/E50-20240626-004339/best.pth.tar\
#     results/teacher/cifar100/tf_efficientnetv2_m/FINETUNE/E50-20240626-013339/best.pth.tar
# )

# for i in $(seq 0 $((${#teachers[@]} - 1)));do
#     teacher_model=${teachers[$i]}
#     resume_path=${paths[$i]}
#     python testTeacher_main.py \
#         --name $name \
#         --model_name $teacher_model \
#         --resume_path $resume_path \
#         --dataset $dataset\
#         --seed $seed
# done