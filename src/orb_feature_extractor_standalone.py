"""
本脚本用于检测图像中的ORB特征，将这些特征均匀地分布在图像上，并保存结果

用法：
    python src/orb_feature_extractor_standalone.py <path_to_images> --output_dir <output_directory>

参数：
    path_to_images: 图像文件或包含图像文件的目录的路径 支持的图像格式为：.jpg、.jpeg、.png、.bmp
    output_dir: (可选) 输出目录 如果未提供，则默认为 "results/ORBEresult"

输出：
    对于处理的每张图像，都会在指定的输出目录中创建一个目录，该目录的名称与图像的名称（不带扩展名）相同
    每个图像目录包含：
        - <image_name>_with_keypoints.jpg：原始图像，并在其上绘制了分布的特征点
        - <image_name>_keypoints_only.png：一个透明的PNG图像，仅包含分布的特征点
        - <image_name>_original.jpg：原始图像的副本
        - info.txt：一个文本文件，其中包含使用的参数以及在分布前后提取的特征点数量

可以在脚本中调整的参数：
    NUM_ORB_FEATURES：最初要检测的ORB特征的最大数量 (默认: 5000)
    NUM_DESIRED_KEYPOINTS：分布后所需的特征点数量 (默认: 500)
    GRID_SIZE：用于分布特征点的网格大小（行数，列数） (默认: (50, 50))
"""

import cv2
import numpy as np
import os
import argparse

# 参数配置
NUM_ORB_FEATURES = 5000
NUM_DESIRED_KEYPOINTS = 500
GRID_SIZE = (50, 50)


def distribute_keypoints(keypoints, image_width, image_height, num_desired_keypoints, grid_size=(4, 4)):
    """
    均匀化分布特征点

    Args:
        keypoints: 原始特征点列表
        image_width: 图像宽度
        image_height: 图像高度
        num_desired_keypoints: 期望的特征点数量
        grid_size: 将图像划分的网格大小 (行数, 列数)

    Returns:
        均匀分布后的特征点列表
    """

    grid_rows, grid_cols = grid_size
    grid_width = image_width // grid_cols
    grid_height = image_height // grid_rows

    distributed_keypoints = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            # 提取当前网格内的特征点
            grid_keypoints = [
                kp for kp in keypoints
                if (j * grid_width <= kp.pt[0] < (j + 1) * grid_width) and
                   (i * grid_height <= kp.pt[1] < (i + 1) * grid_height)
            ]

            # 如果网格内有特征点，则保留响应值最高的特征点
            if grid_keypoints:
                best_keypoint = max(grid_keypoints, key=lambda kp: kp.response)
                distributed_keypoints.append(best_keypoint)

    # 如果特征点数量不足，可以考虑调整网格大小或者增加特征点检测的数量
    if len(distributed_keypoints) < num_desired_keypoints:
        print(f"Warning: Found only {len(distributed_keypoints)} keypoints, "
              f"less than the desired {num_desired_keypoints}.")

    return distributed_keypoints

def process_image(image_path, output_dir):
    """
    处理单张图片，检测ORB特征，均匀化分布，并保存结果

    Args:
        image_path: 图片路径
        output_dir: 输出目录
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_dir = os.path.join(output_dir, image_name)
    os.makedirs(output_image_dir, exist_ok=True)

    # ORB特征检测
    orb = cv2.ORB_create(nfeatures=NUM_ORB_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    num_extracted_keypoints = len(keypoints)
    print(f"Extracted {num_extracted_keypoints} keypoints initially.")

    # 特征点均匀化分布
    distributed_keypoints = distribute_keypoints(keypoints, img.shape[1], img.shape[0], NUM_DESIRED_KEYPOINTS, GRID_SIZE)


    # 绘制特征点到原图
    img_with_keypoints = cv2.drawKeypoints(img, distributed_keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imwrite(os.path.join(output_image_dir, f"{image_name}_with_keypoints.jpg"), img_with_keypoints)

    # 创建只包含特征点的透明底png图
    img_keypoints_only = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)  # RGBA
    for kp in distributed_keypoints:
        x, y = map(int, kp.pt)
        cv2.circle(img_keypoints_only, (x, y), radius=3, color=(0, 255, 0, 255), thickness=-1)  # 绿色，完全不透明
    cv2.imwrite(os.path.join(output_image_dir, f"{image_name}_keypoints_only.png"), img_keypoints_only)

    # 保存原图
    cv2.imwrite(os.path.join(output_image_dir, f"{image_name}_original.jpg"), img)

    # 保存参数和特征点数量到txt文件
    with open(os.path.join(output_image_dir, "info.txt"), "w") as f:
        f.write(f"Image Path: {image_path}\n")
        f.write("Parameters:\n")
        f.write(f"  NUM_ORB_FEATURES: {NUM_ORB_FEATURES}\n")
        f.write(f"  NUM_DESIRED_KEYPOINTS: {NUM_DESIRED_KEYPOINTS}\n")
        f.write(f"  GRID_SIZE: {GRID_SIZE}\n")
        f.write("\n")
        f.write("Results:\n")
        f.write(f"  Number of extracted keypoints (before distribution): {num_extracted_keypoints}\n")
        f.write(f"  Number of keypoints after distribution: {len(distributed_keypoints)}\n")


    print(f"Processed image: {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect ORB features in images, distribute them evenly, and save results.")
    parser.add_argument("path_to_images", help="Path to the image file or directory containing images.")
    parser.add_argument("--output_dir", help="Path to the output directory. Defaults to 'results/ORBEresult'.", default="results/ORBEresult")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(args.path_to_images):
        process_image(args.path_to_images, output_dir)
    elif os.path.isdir(args.path_to_images):
        for filename in os.listdir(args.path_to_images):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 检查文件类型
                image_path = os.path.join(args.path_to_images, filename)
                process_image(image_path, output_dir)
    else:
        print(f"Error: {args.path_to_images} is not a valid file or directory.")

if __name__ == "__main__":
    main()
