#!/bin/bash

# 用法：
# ./run.sh -p workfolder/images/1.jpg -b

# -b 小  -l 中  -h 大
# -p 输入图像路径

# 默认值
SAM_VERSION="vit_l"
SAM_CHECKPOINT="../models/sam_vit_l_0b3195.pth"
INPUT_IMAGE=""
BOX_THRESHOLD=0.3
TEXT_THRESHOLD=0.25
TEXT_PROMPT="car . truck . motorbike . bicycle . people"
DEVICE="cuda"
OUTPUT_DIR=""

# 函数：提取不带扩展名的文件名
get_filename_without_extension() {
  local filepath="$1"
  local filename=$(basename "$filepath")
  echo "${filename%.*}"
}

# 解析命令行参数
while getopts "hlbp:" opt; do
  case "$opt" in
    h)
      SAM_VERSION="vit_h"
      SAM_CHECKPOINT="../models/sam_vit_h_4b8939.pth" # 替换为实际的 vit_h 检查点名称
      ;;
    l)
      SAM_VERSION="vit_l"
      SAM_CHECKPOINT="../models/sam_vit_l_0b3195.pth"
      ;;
    b)
      SAM_VERSION="vit_b"
      SAM_CHECKPOINT="../models/sam_vit_b_01ec64.pth" # 替换为实际的 vit_b 检查点名称
      ;;
    p)
      INPUT_IMAGE="$OPTARG"
      ;;
    \?)
      echo "无效选项: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "选项 -$OPTARG 需要一个参数。" >&2
      exit 1
      ;;
  esac
done

# 检查是否提供了输入图像
if [ -z "$INPUT_IMAGE" ]; then
  echo "错误：需要输入图像路径。使用 -p <图像路径>"
  exit 1
fi

# 设置输出目录
IMAGE_NAME=$(get_filename_without_extension "$INPUT_IMAGE")
OUTPUT_DIR="../workfolder/GSAresults/${IMAGE_NAME}_${SAM_VERSION}"
OUTPUT_OUTPUT_DIR="workfolder/GSAresults/${IMAGE_NAME}_${SAM_VERSION}"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 运行 Python 脚本
cd Grounded-Segment-Anything

export CUDA_VISIBLE_DEVICES=0

python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ../models/groundingdino_swint_ogc.pth \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --sam_version "$SAM_VERSION" \
  --input_image "../$INPUT_IMAGE" \
  --output_dir "$OUTPUT_DIR" \
  --box_threshold "$BOX_THRESHOLD" \
  --text_threshold "$TEXT_THRESHOLD" \
  --text_prompt "$TEXT_PROMPT" \
  --device "$DEVICE"

echo "处理完成。结果保存在：$OUTPUT_OUTPUT_DIR"
