#!/bin/bash
type=$1
arch=$2
if [ ${arch} = "cell" ]; then
    if [ ${type} = "train" ]; then
        # ===== セルレベルアーキテクチャを評価　=====
        name=$3
        teacher_model=$4
        teacher_path=$5
        genotype=$6
        save=$7
        description=$8
        dataset=cifar100
        lambda=$9
        T=${10}
        batch_size=64
        epoch=100
        train_portion=0.9
        seed=${11}
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
            --description $description \
            --advanced \
            --nonkd
    elif [ ${type} = "test" ]; then
        ## セルレベル構造のテスト
        genotype=$3
        path=$4
        seed=0

        python testModel_main.py \
            --save test \
            --dataset cifar100 \
            --batch_size 128 \
            --genotype $genotype \
            --seed $seed \
            --resume_path $path \
            --advanced
    else
        echo "Invalid arguments"
    fi
elif [ ${arch} = "stage" ]; then
    if [ ${type} = "train" ]; then
        # ===== セルレベルアーキテクチャを評価　=====
        name=$3
        teacher_model=$4
        teacher_path=$5
        genotype=$6
        dag=$7
        save=$8
        description=$9
        dataset=cifar10
        lambda=0.5
        T=10
        epoch=100
        train_portion=0.9
        seed=${10}
        python evaluateStage_main.py \
            --name $name \
            --genotype $genotype \
            --DAG $dag \
            --teacher_name $teacher_model\
            --teacher_path $teacher_path \
            --l $lambda\
            --T $T \
            --dataset $dataset \
            --epochs $epoch \
            --train_portion $train_portion \
            --seed $seed \
            --save $save \
            --description $description \
            --spec_cell \
            --nonkd
            # --advanced \
    elif [ ${type} = "test" ]; then
        ## セルレベル構造のテスト
        genotype=$3
        dag=$4
        path=$5
        seed=0

        python testModel_main.py \
            --save test \
            --dataset cifar100 \
            --batch_size 128 \
            --genotype $genotype \
            --DAG $dag \
            --seed $seed \
            --resume_path $path \
            --stage \
            --spec_cell \
            --advanced
    else
        echo ""
    fi
fi
