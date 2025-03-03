#!/bin/bash

# 检查是否提供了图像路径作为参数
if [ -z "$1" ]; then
  echo "Error: 请提供图像的路径作为参数。"
  echo "Usage: ./your_script_name.sh <path_to_image>"
  exit 1
fi

path_to_image="$1"

# 从路径中提取文件名
image_filename=$(basename "$path_to_image")

# 从文件名中移除扩展名
image_name_without_extension="${image_filename%.*}"

# 构建输出路径
output_path="results/DINOresults/${image_name_without_extension}"

# 运行 GroundingDINO 脚本
CUDA_VISIBLE_DEVICES=0 python GroundingDINO/demo/inference_on_a_image.py \
-c GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p models/groundingdino_swint_ogc.pth \
-i "$path_to_image" \
-o "$output_path" \
-t "car . truck . motorbike . bicycle . people"

echo "推理完成。结果保存到: $output_path"
