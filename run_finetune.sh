#!/bin/bash

type=$1
if [ ${type} = "train" ]; then
# ===== モデルをファインチューニング =====
        name=$2
        teacher_model=$3
        save=$4
        description=$5
        dataset=cifar100
        cutout=16
        epoch=100
        seed=0
        train_portion=0.9
        python trainTeacher_main.py \
            --name $name \
            --model_name $teacher_model\
            --dataset $dataset\
            --cutout_length $cutout\
            --epochs $epoch \
            --seed $seed \
            --save $save \
            --train_portion $train_portion \
            --lr 0.1 \
            --lr_min 0.001 \
            --description $description \
            # --advanced \
            # --pretrained
elif [ ${type} = "test" ]; then
# ===== モデルをテスト =====
        save=$2
        teacher_model=$3
        resume_path=$4
        dataset=cifar100
        cutout=0
        batch_size=64
        seed=0
        train_portion=1.0
        python testTeacher_main.py \
                --save $save \
                --model_name $teacher_model \
                --resume_path $resume_path \
                --dataset $dataset\
                --seed $seed \
                --advanced
else
    echo ""
fi

## ===== 複数モデルをファインチューニングする =====
# name=$1
# dataset=cifar100
# cutout=0
# batch_size=64
# epoch=100
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

## ===== 複数モデルをテスト =====
# save=$1
# dataset=cifar100
# seed=0
# teachers=("densenet121" "densenet161" "resnet50" "resnet152" "wide_resnet50_2" "tf_efficientnetv2_s" "tf_efficientnetv2_m")
# paths=(
#     /home/miura/lab/KD-hdas/results/teacher/cifar100/densenet121/FINETUNE/E200-20240627-060151/best.pth.tar\
#     /home/miura/lab/KD-hdas/results/teacher/cifar100/densenet161/FINETUNE/E200-20240627-090330/best.pth.tar\
#     /home/miura/lab/KD-hdas/results/teacher/cifar100/resnet50/FINETUNE/E200-20240627-130656/best.pth.tar\
#     /home/miura/lab/KD-hdas/results/teacher/cifar100/resnet152/FINETUNE/E200-20240627-144917/best.pth.tar\
#     /home/miura/lab/KD-hdas/results/teacher/cifar100/wide_resnet50_2/FINETUNE/E200-20240627-060143/best.pth.tar\
#     /home/miura/lab/KD-hdas/results/teacher/cifar100/tf_efficientnetv2_s/FINETUNE/E200-20240627-074940/best.pth.tar\
#     /home/miura/lab/KD-hdas/results/teacher/cifar100/tf_efficientnetv2_m/FINETUNE/E200-20240627-111054/best.pth.tar
# )

# for i in $(seq 0 $((${#teachers[@]} - 1)));do
#     teacher_model=${teachers[$i]}
#     resume_path=${paths[$i]}
#     python testTeacher_main.py \
#         --save $save \
#         --model_name $teacher_model \
#         --resume_path $resume_path \
#         --dataset $dataset\
#         --seed $seed
# done

# ===== モデルをファインチューニング ===== 廃止
# name=$1

# teacher_model=$2
# save=$3
# dataset=cifar100
# cutout=16
# batch_size=64
# epoch=300
# seed=0
# train_portion=0.9
# python finetuneTeacher_main.py \
#     --name $name \
#     --model_name $teacher_model\
#     --dataset $dataset\
#     --cutout_length $cutout\
#     --batch_size $batch_size \
#     --epochs $epoch \
#     --seed $seed \
#     --save $save \
#     --train_portion $train_portion \
#     --w_lr 0.1 \
#     --w_grad_clip 100.
