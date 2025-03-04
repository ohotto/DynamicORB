#!/bin/bash

# 检查是否提供了两个图像路径作为参数
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: 请提供两个图像的路径作为参数。"
  echo "Usage: ./scripts/runORBfm.sh <path_to_image1> <path_to_image2>"
  exit 1
fi

path_to_image1="$1"
path_to_image2="$2"

# 从路径中提取文件名
image1_filename=$(basename "$path_to_image1")
image2_filename=$(basename "$path_to_image2")

# 从文件名中移除扩展名
image1_name_without_extension="${image1_filename%.*}"
image2_name_without_extension="${image2_filename%.*}"

# 构建输出路径
output_path="results/ORBMresult/${image1_name_without_extension}-${image2_name_without_extension}"

# 运行 orb_feature_matcher 脚本
python src/orb_feature_matcher.py "$path_to_image1" "$path_to_image2"

echo "匹配完成。结果保存到: $output_path"
