target_directory="/home/miura/lab/KD-hdas/results/search_stage_KD/cifar100/noDepthLoss/s0-BaselineBestCell/DAG"

# ディレクトリ内で「best」を含むファイルをリストアップし、最も新しいファイルを見つける
newest_file=$(find "$target_directory" -type f -name '*best*' -exec stat --format="%Y %n" {} + | sort -nr | head -n 1 | awk '{print $2}')
echo $newest_file