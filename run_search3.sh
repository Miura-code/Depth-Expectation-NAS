#!/bin/bash

method=$1
type=$2
arch=$3
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
        name=$4
        teacher_model=$5
        teacher_path=$6
        save=$7
        genotype=$8
        description=$9
        seed=${10}
        dataset=cifar100
        lambda=${11}
        min_lambda=${12}
        T=10
        batch_size=64
        epoch=50
        eval_epoch=100
        train_portion=0.5 # searchの場合train_portionは0.5が最大値
        python searchStage_main.py \
            --type $method \
            --name $name \
            --genotype $genotype \
            --teacher_name $teacher_model\
            --teacher_path $teacher_path \
            --l $lambda\
            --final_l $min_lambda \
            --T $T \
            --dataset $dataset\
            --batch_size $batch_size \
            --epochs $epoch \
            --eval_epochs $eval_epoch \
            --train_portion $train_portion \
            --seed $seed \
            --save $save \
            --spec_cell \
            --description $description \
            --depth_coef 0 \
            --slide_window 3 \
            --advanced \
            --nonkd
            # --cascade
            # --pcdarts
    elif [ ${type} = "test" ]; then
        # ===== モデルをテスト =====
        resume_path=$4
        genotype=$5
        dag=$6
        save=test
        dataset=cifar100
        cutout=0
        batch_size=64
        seed=0
        train_portion=1.0
        python testSearchedModel_main.py \
            --save $save \
            --resume_path $resume_path \
            --genotype $genotype \
            --DAG $dag \
            --dataset $dataset\
            --batch_size $batch_size \
            --train_portion $train_portion \
            --seed $seed \
            --spec_cell \
            --slide_window 8 \
            --advanced
    else
        echo ""
    fi
fi




