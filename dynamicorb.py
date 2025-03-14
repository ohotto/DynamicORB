"""
动态ORB特征点剔除程序

使用说明：
该程序用于提取图像的ORB特征点，并结合实例分割结果剔除动态特征点

运行方式：
python dynamicorb.py --input <input_image_path> --output <output_directory>

参数：
--input: 输入图像路径
--output: 输出目录路径

程序会：
1. 提取图像的ORB特征点
2. 使用 Grounded-SAM 模型对图像进行实例分割
3. 剔除动态特征点，生成结果可视化
4. 输出处理结果报告

配置参数：
- ORB_CONFIG: ORB特征提取参数
- SEGMENT_CONFIG: 实例分割参数
- MODEL_PATHS: 模型路径配置
- OUTPUT_CONFIG: 输出配置

输出：
- static_orb_overlay.jpg: 叠加静态特征点图
- static_orb_transparent.png: 透明背景静态特征点图
- result_report.txt: 处理结果报告
- mask_results/: 实例分割结果目录
    - GSAoutput.jpg: 包含原始图像、检测框和分割掩膜的可视化结果
    - GSAmask.jpg: 分割掩膜的可视化图像，每个实例用不同的颜色表示
    - GSAmask.json: 包含每个实例的标签、置信度得分和边界框信息的 JSON 文件
    - detection_boxes_transparent.png: 带有透明检测框的图像
    - segmentation_mask_transparent.png: 透明背景的分割掩膜
    - dilated_mask_transparent.png: 经过形态学膨胀处理的透明掩膜
    - processed_mask.jpg: 最终处理后的掩膜图像
    - PMASKinfo.txt: 包含处理参数和统计信息的文本文件
- orb_results/: ORB特征提取结果目录
    - original.jpg: 原始图像
    - with_keypoints.jpg: 带特征点的图像
    - keypoints_only.png: 仅特征点的图像
    - ORBinfo.txt: 包含参数配置和特征点数量的信息文件
"""
import sys
import os
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from process_mask import MaskProcessor
from orb_feature_extractor import ORBFeatureExtractor

def main():
    # ====================== 参数配置区 ======================
    # ORB特征提取参数
    ORB_CONFIG = {
        'num_orb_features': 5000,     # ORB最大特征点数
        'num_desired_keypoints': 500,  # 期望特征点数
        'grid_size': (50, 50)         # 分布网格(行,列)
    }
    
    # 实例分割参数
    SEGMENT_CONFIG = {
        'text_prompt': "car . truck . motorbike . bicycle . people",  # 检测目标类别
        'box_threshold': 0.3,        # 检测框置信度阈值
        'text_threshold': 0.25,      # 文本匹配阈值
        'dilation_kernel': 5,        # 膨胀卷积核大小
        'dilation_iters': 2          # 膨胀迭代次数
    }
    
    # 模型路径配置
    MODEL_PATHS = {
        'grounded_config': "./configs/GroundingDINO_SwinT_OGC.py",
        'grounded_checkpoint': "./models/groundingdino_swint_ogc.pth",
        'sam_checkpoint': "./models/sam_vit_l_0b3195.pth"
    }
    
    # 输出配置
    OUTPUT_CONFIG = {
        'overlay_color': (0, 255, 0),   # 特征点颜色(BGR)
        'point_radius': 3,             # 点半径
        'transparent_alpha': 255        # 透明图Alpha值
    }
    # =======================================================

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='动态ORB特征点剔除')
    parser.add_argument('--input', required=True, help='输入图像路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    args = parser.parse_args()

    # 创建输出目录结构
    orb_output = os.path.join(args.output, "orb_results")
    mask_output = os.path.join(args.output, "mask_results")
    os.makedirs(orb_output, exist_ok=True)
    os.makedirs(mask_output, exist_ok=True)

    # 阶段1: ORB特征提取
    print("\n=== 正在提取ORB特征 ===")
    orb_extractor = ORBFeatureExtractor(**ORB_CONFIG)
    orb_result = orb_extractor.extract(args.input)
    orb_extractor.process_and_save(args.input, orb_output)
    
    # 阶段2: 动态区域分割与膨胀
    print("\n=== 正在处理动态区域蒙版 ===")
    mask_processor = MaskProcessor(
        config_path=MODEL_PATHS['grounded_config'],
        grounded_checkpoint=MODEL_PATHS['grounded_checkpoint'],
        sam_checkpoint=MODEL_PATHS['sam_checkpoint']
    )
    mask_result = mask_processor.process(
        args.input, mask_output,
        text_prompt=SEGMENT_CONFIG['text_prompt'],
        box_threshold=SEGMENT_CONFIG['box_threshold'],
        text_threshold=SEGMENT_CONFIG['text_threshold'],
        dilation_kernel_size=SEGMENT_CONFIG['dilation_kernel'],
        dilation_iterations=SEGMENT_CONFIG['dilation_iters']
    )

    # 阶段3: 动态特征点剔除
    print("\n=== 正在剔除动态特征点 ===")
    # 加载膨胀后的蒙版
    mask = cv2.imread(mask_result['final_mask_path'], cv2.IMREAD_GRAYSCALE)
    
    # 过滤动态特征点
    static_keypoints = []
    dynamic_count = 0
    for kp in orb_result['keypoints']:
        x, y = map(int, kp.pt)
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if mask[y, x] < 127:  # 蒙版值<127视为静态区域
                static_keypoints.append(kp)
            else:
                dynamic_count += 1
        else:
            static_keypoints.append(kp)  # 超出图像范围的点保留

    # 生成结果可视化
    print("\n=== 生成结果可视化 ===\n")
    # 叠加静态点的原图
    overlay_img = cv2.drawKeypoints(
        orb_result['original_image'], static_keypoints, None,
        color=OUTPUT_CONFIG['overlay_color'],
        # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        flags=0
    )
    
    # 透明背景特征点图
    transparent_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    for kp in static_keypoints:
        x, y = map(int, kp.pt)
        cv2.circle(transparent_img, (x, y),
                   OUTPUT_CONFIG['point_radius'],
                   (*OUTPUT_CONFIG['overlay_color'], OUTPUT_CONFIG['transparent_alpha']),
                   -1)

    # 保存最终结果
    final_output = {
        'overlay': os.path.join(args.output, "static_orb_overlay.jpg"),
        'transparent': os.path.join(args.output, "static_orb_transparent.png"),
        'report': os.path.join(args.output, "result_report.txt")
    }
    
    cv2.imwrite(final_output['overlay'], overlay_img)
    cv2.imwrite(final_output['transparent'], transparent_img)

    # 读取附加信息文件
    orb_info_path = os.path.join(args.output, "orb_results", "ORBinfo.txt")
    pmask_info_path = os.path.join(args.output, "mask_results", "PMASKinfo.txt")

    def read_info_file(path):
        try:
            with open(path, 'r') as f:
                return f.read().rstrip('\n')
        except FileNotFoundError:
            return f"警告：文件 {os.path.basename(path)} 未找到"
        
    orb_info = read_info_file(orb_info_path)
    pmask_info = read_info_file(pmask_info_path)

    # 计算动态特征点占比
    total_features = len(orb_result['keypoints'])
    dynamic_ratio = dynamic_count / total_features if total_features > 0 else 0.0
    
    # 生成统计报告
    report_content = f"""=== ORB特征提取 ===
{orb_info}

=== 动态区域分割 ===
{pmask_info}
    
=== 处理结果报告 ===
输入图像: {args.input}
总提取特征点数: {total_features}
静态特征点数: {len(static_keypoints)}
剔除动态特征点数: {dynamic_count}
动态特征点占比: {dynamic_ratio:.2%}
动态区域占比: {mask_result['area_ratio']:.2%}
"""
    with open(final_output['report'], 'w') as f:
        f.write(report_content)
    
    print(report_content)
    print(f"处理结果已保存至: {args.output}")

if __name__ == "__main__":
    main()
