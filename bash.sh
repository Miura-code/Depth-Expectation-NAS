#!/bin/bash

# 操作するディレクトリを指定（カレントディレクトリの場合は '.'）
TARGET_DIR="results/search_stage_KD/cifar100/SEARCHEVALnoDL"

# 対象の末尾文字列
SUFFIX="relaxEval"

# 指定ディレクトリ内のサブディレクトリを探索
for dir in "$TARGET_DIR"/*; do
  if [ -d "$dir" ] && [[ "${dir##*/}" == *"$SUFFIX" ]]; then
    new_name="${dir}-L20"
    mv "$dir" "$new_name"
    echo "Renamed: $dir -> $new_name"
  fi
done
