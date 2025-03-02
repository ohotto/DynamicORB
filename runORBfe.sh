#!/bin/bash

# 检查是否提供了图像路径作为参数
if [ -z "$1" ]; then
  echo "Error: 请提供图像的路径作为参数。"
  echo "Usage: ./runORBfe.sh <path_to_image>"
  exit 1
fi

path_to_image="$1"

# 从路径中提取文件名
image_filename=$(basename "$path_to_image")

# 从文件名中移除扩展名
image_name_without_extension="${image_filename%.*}"

# 构建输出路径
output_path="workfolder/ORBFEresults/${image_name_without_extension}"

# 运行 runORBfe 脚本
python src/orb_feature_extractor.py $path_to_image

echo "提取完成。结果保存到: $output_path"
