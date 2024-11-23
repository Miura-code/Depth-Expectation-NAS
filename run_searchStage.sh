#!/bin/bash

type=$1
method=$2

if [ ${type} = "train" ]; then
    # ===== ステージレベル探索　=====
    dataset=cifar100
    T=10
    batch_size=64
    epoch=50
    eval_epoch=100
    train_portion=0.5 # searchの場合train_portionは0.5が最大値
    macs="8.18 2.49 1.88"
    layer=20

    name=$3
    teacher_model=$4
    teacher_path=$5
    save=$6
    genotype=$7
    description=$8
    seed=${9}
    lambda=${10}
    min_lambda=${11}
    gamma=${12}
    nonkd=${13}
    depth_coef=${14}
    slide_window=${15}
    discrete=${16}
    reset=${17}
    arch_criterion=${18}
    curri_epoch=${19}
    
    python searchStage_main.py \
        --type $method \
        --name $name \
        --save $save \
        --dataset $dataset\
        --batch_size $batch_size \
        --train_portion $train_portion \
        --advanced 1\
        --epochs $epoch \
        --eval_epochs $eval_epoch \
        --curriculum_epochs $curri_epoch \
        --T $T \
        --l $lambda\
        --g $gamma\
        --final_l $min_lambda \
        --seed $seed \
        --nonkd $nonkd\
        --depth_coef $depth_coef \
        --slide_window $slide_window \
        --discrete $discrete\
        --reset $reset\
        --arch_criterion $arch_criterion\
        --stage_macs $macs \
        --spec_cell 1\
        --layers $layer \
        --genotype $genotype \
        --teacher_name $teacher_model\
        --teacher_path $teacher_path \
        --description $description
        # --cascade
        # --pcdarts
elif [ ${type} = "test" ]; then
    # ===== モデルをテスト =====
    save=test
    dataset=cifar100
    cutout=0
    batch_size=64
    seed=0
    train_portion=1.0
    layer=32
    resume_path=$3
    genotype=$4
    dag=$5
    slide_window=${6}
    discrete=${7}

    python testSearchedModel_main.py \
        --type $method \
        --save $save \
        --resume_path $resume_path \
        --genotype $genotype \
        --DAG $dag \
        --dataset $dataset\
        --batch_size $batch_size \
        --train_portion $train_portion \
        --seed $seed \
        --spec_cell 1\
        --layers $layer \
        --slide_window $slide_window \
        --advanced 1\
        --discrete $discrete
else
    echo ""
fi