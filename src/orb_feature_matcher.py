"""
ORB特征提取与匹配程序

用法：
python orb_feature_matching.py <path_to_image1> <path_to_image2>

配置参数说明：
[ORB参数]
ORB_MAX_FEATURES = 1000      # 最大特征检测数量（0表示无限制）
ORB_SCALE_FACTOR = 1.2       # 金字塔缩放因子
ORB_N_LEVELS = 8             # 金字塔层数
ORB_EDGE_THRESHOLD = 31      # 边缘阈值
[均匀分布参数]
GRID_ROWS = 5                # 分布网格行数
GRID_COLS = 5                # 分布网格列数
MAX_PER_GRID = 20            # 每个网格最大特征点数
[匹配参数]
MATCH_RATIO_THRESH = 0.75    # 匹配质量阈值
[路径参数]
RESULT_ROOT = "results/ORBMresult"  # 结果保存路径
"""

import cv2
import argparse
import os
import numpy as np
from collections import defaultdict

# 配置参数
# ORB参数
ORB_MAX_FEATURES = 1500
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_EDGE_THRESHOLD = 10
# 均匀分布参数
GRID_ROWS = 10
GRID_COLS = 10
MAX_PER_GRID = 100
# 匹配参数
MATCH_RATIO_THRESH = 0.6
# 路径参数
RESULT_ROOT = "results/ORBMresult"

def uniform_distribute_keypoints(keypoints, descriptors, img_shape, grid_rows, grid_cols, max_per_grid):
    """均匀分布关键点并返回对应描述符"""
    height, width = img_shape[:2]
    grid_w = width / grid_cols
    grid_h = height / grid_rows
    
    grid = defaultdict(list)
    for idx, kp in enumerate(keypoints):
        x, y = kp.pt
        col = min(int(x // grid_w), grid_cols - 1)
        row = min(int(y // grid_h), grid_rows - 1)
        grid[(row, col)].append( (kp, idx) )  # 保存关键点和原始索引
    
    selected_kp = []
    selected_idx = []
    for cell in grid.values():
        sorted_cell = sorted(cell, key=lambda x: -x[0].response)[:max_per_grid]
        selected_kp.extend([kp for kp, idx in sorted_cell])
        selected_idx.extend([idx for kp, idx in sorted_cell])
    
    # 返回筛选后的关键点和对应的描述符
    return selected_kp, descriptors[selected_idx] if descriptors is not None else None

def draw_transparent_keypoints(img_shape, keypoints):
    """绘制透明背景关键点"""
    h, w = img_shape[:2]
    transparent = np.zeros((h, w, 4), dtype=np.uint8)
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(transparent, (int(x), int(y)), 5, (0, 255, 0, 255), -1)
    return transparent

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('path1', help="第一张图片路径")
    parser.add_argument('path2', help="第二张图片路径")
    args = parser.parse_args()

    # 读取并校验图片
    img1 = cv2.imread(args.path1)
    img2 = cv2.imread(args.path2)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图片，请检查路径是否正确")

    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化ORB检测器
    orb = cv2.ORB_create(
        nfeatures=ORB_MAX_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS,
        edgeThreshold=ORB_EDGE_THRESHOLD
    )

    # 检测关键点和描述符
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 均匀分布处理（带描述符筛选）
    kp1_uni, des1_uni = uniform_distribute_keypoints(kp1, des1, gray1.shape, GRID_ROWS, GRID_COLS, MAX_PER_GRID)
    kp2_uni, des2_uni = uniform_distribute_keypoints(kp2, des2, gray2.shape, GRID_ROWS, GRID_COLS, MAX_PER_GRID)

    # 特征匹配（使用筛选后的描述符）
    good_matches = []
    if des1_uni is not None and des2_uni is not None and len(des1_uni) > 0 and len(des2_uni) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1_uni, des2_uni, k=2)
        for m, n in matches:
            if m.distance < MATCH_RATIO_THRESH * n.distance:
                good_matches.append(m)

    # 准备输出信息
    output = [
        "本次运行的配置参数：",
        f"ORB_MAX_FEATURES = {ORB_MAX_FEATURES}",
        f"ORB_SCALE_FACTOR = {ORB_SCALE_FACTOR}",
        f"ORB_N_LEVELS = {ORB_N_LEVELS}",
        f"ORB_EDGE_THRESHOLD = {ORB_EDGE_THRESHOLD}",
        f"GRID_ROWS = {GRID_ROWS}",
        f"GRID_COLS = {GRID_COLS}",
        f"MAX_PER_GRID = {MAX_PER_GRID}",
        f"MATCH_RATIO_THRESH = {MATCH_RATIO_THRESH}",
        f"\n图片1提取特征点数量：{len(kp1_uni)}",
        f"图片2提取特征点数量：{len(kp2_uni)}",
        f"匹配的特征点数量：{len(good_matches)}"
    ]

    # 创建保存目录
    img_name1 = os.path.splitext(os.path.basename(args.path1))[0]
    img_name2 = os.path.splitext(os.path.basename(args.path2))[0]
    save_dir = os.path.join(RESULT_ROOT, f"{img_name1}-{img_name2}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存结果图片
    cv2.imwrite(os.path.join(save_dir, f"{img_name1}.jpg"), img1)
    cv2.imwrite(os.path.join(save_dir, f"{img_name1}_keypoints.jpg"), 
                cv2.drawKeypoints(img1, kp1_uni, None, color=(0, 255, 0)))
    cv2.imwrite(os.path.join(save_dir, f"{img_name1}_transparent.png"), 
                draw_transparent_keypoints(gray1.shape, kp1_uni))

    cv2.imwrite(os.path.join(save_dir, f"{img_name2}.jpg"), img2)
    cv2.imwrite(os.path.join(save_dir, f"{img_name2}_keypoints.jpg"), 
                cv2.drawKeypoints(img2, kp2_uni, None, color=(0, 255, 0)))
    cv2.imwrite(os.path.join(save_dir, f"{img_name2}_transparent.png"), 
                draw_transparent_keypoints(gray2.shape, kp2_uni))

    # 绘制并保存匹配结果
    if len(good_matches) > 0:
        match_img = cv2.drawMatches(img1, kp1_uni, img2, kp2_uni, good_matches, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.path.join(save_dir, "matches.jpg"), match_img)

    # 保存并输出结果信息
    with open(os.path.join(save_dir, "info.txt"), "w") as f:
        for line in output:
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    main()
