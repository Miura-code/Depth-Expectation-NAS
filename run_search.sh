#!/bin/bash

type=$1
arch=$2
if [ ${arch} = "cell" ]; then
    if [ ${type} = "train" ]; then
    # ===== セルレベル探索　=====
        name=$3
        teacher_model=$4
        teacher_path=$5
        save=$6
        description=$7
        dataset=cifar100
        lambda=0.5
        T=10
        batch_size=64
        epoch=50
        train_portion=0.5 # searchの場合train_portionは0.5が最大値
        seed=$8
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
            --save $save \
            --advanced \
            --description $description
            # --nonkd
            # --pcdarts
    elif [ ${type} = "test" ]; then
        # ===== モデルをテスト =====
        resume_path=$3
        genotype=$4
        save=test
        dataset=cifar100
        cutout=0
        batch_size=64
        seed=0
        train_portion=1.0
        python testModel_main.py \
                --save $save \
                --resume_path $resume_path \
                --genotype $genotype \
                --dataset $dataset\
                --seed $seed
    else
        echo ""
    fi
elif [ ${arch} = "stage" ]; then
    if [ ${type} = "train" ]; then
    # ===== ステージレベル探索　=====
        name=$3
        teacher_model=$4
        teacher_path=$5
        save=$6
        genotype=$7
        description=$8
        dataset=cifar100
        lambda=0.6
        T=3
        batch_size=64
        epoch=50
        train_portion=0.5 # searchの場合train_portionは0.5が最大値
        seed=0
        python searchStage_KD_main.py \
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
            --spec_cell \
            --advanced \
            --description $description\
            --nonkd \
            --pcdarts
    elif [ ${type} = "test" ]; then
        # ===== モデルをテスト =====
        resume_path=$3
        genotype=$4
        dag=$5
        save=test
        dataset=cifar100
        cutout=0
        batch_size=64
        seed=0
        train_portion=1.0
        python testModel_main.py \
                --save $save \
                --resume_path $resume_path \
                --genotype $genotype \
                --DAG $dag \
                --dataset $dataset\
                --seed $seed
    else
        echo ""
    fi
fi




