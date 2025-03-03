"""
# 处理单张图像
python src/orb_feature_extractor1.py images/1.jpg --output_dir test --num_orb 5000 --num_desired 500 --grid_rows 50 --grid_cols 50
# 处理目录下所有图像
python src/orb_feature_extractor.py images/ -o my_results --grid_rows 30 --grid_cols 30
"""
import cv2
import numpy as np
import os
import argparse

class ORBFeatureExtractor:
    """
    ORB特征提取器，支持特征点检测和均匀分布，可配置参数并输出结果图像。
    
    参数:
        num_orb_features (int): ORB特征的最大数量，默认5000。
        num_desired_keypoints (int): 期望的均匀分布特征点数，默认500。
        grid_size (tuple): 分布特征点的网格尺寸(行数, 列数)，默认(50, 50)。
    """
    
    def __init__(self, num_orb_features=5000, num_desired_keypoints=500, grid_size=(50, 50)):
        if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
            raise ValueError("grid_size must be a tuple of two integers (rows, cols)")
        self.num_orb_features = num_orb_features
        self.num_desired_keypoints = num_desired_keypoints
        self.grid_size = grid_size  # (rows, cols)
        self.orb = cv2.ORB_create(nfeatures=num_orb_features)

    def extract(self, image):
        """
        提取ORB特征点并均匀分布。
        
        Args:
            image (str/np.ndarray): 输入图像路径或数组。
        
        Returns:
            dict: 包含特征点、描述子、结果图像及数量的字典。
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        else:
            img = image.copy()

        # 检测ORB特征
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        if not keypoints:
            raise ValueError("未检测到任何特征点")

        # 均匀分布特征点
        distributed_kps = self._distribute_keypoints(keypoints, img.shape)

        # 生成结果图像
        img_with_kp = self._draw_keypoints_on_image(img, distributed_kps)
        kp_only_img = self._create_keypoints_only_image(img.shape, distributed_kps)

        return {
            "original_image": img,
            "keypoints": distributed_kps,
            "descriptors": descriptors,
            "image_with_keypoints": img_with_kp,
            "keypoints_only_image": kp_only_img,
            "num_keypoints": len(distributed_kps)
        }

    def _distribute_keypoints(self, keypoints, image_shape):
        """均匀分布特征点到网格中，每个网格保留响应最高的点。"""
        grid_rows, grid_cols = self.grid_size
        img_h, img_w = image_shape[:2]
        grid_w = img_w // grid_cols
        grid_h = img_h // grid_rows

        distributed = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                left = j * grid_w
                right = (j+1) * grid_w
                top = i * grid_h
                bottom = (i+1) * grid_h

                # 筛选当前网格内的特征点
                candidates = [kp for kp in keypoints 
                             if left <= kp.pt[0] < right and top <= kp.pt[1] < bottom]
                if candidates:
                    best_kp = max(candidates, key=lambda kp: kp.response)
                    distributed.append(best_kp)

        if len(distributed) < self.num_desired_keypoints:
            print(f"Warning: 仅分布 {len(distributed)} 个特征点，少于期望的 {self.num_desired_keypoints}。")
        return distributed

    def _draw_keypoints_on_image(self, img, keypoints):
        """在原图上绘制特征点，返回BGR图像。"""
        return cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

    def _create_keypoints_only_image(self, img_shape, keypoints):
        """生成透明背景的特征点图，返回RGBA图像。"""
        img = np.zeros((img_shape[0], img_shape[1], 4), dtype=np.uint8)
        for kp in keypoints:
            x, y = map(int, kp.pt)
            cv2.circle(img, (x, y), 3, (0, 255, 0, 255), -1)
        return img

    def process_and_save(self, image_path, output_dir):
        """
        处理图像并保存结果到指定目录。
        
        Args:
            image_path (str): 输入图像路径。
            output_dir (str): 输出目录路径。
        """
        result = self.extract(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        # 保存原图
        cv2.imwrite(os.path.join(output_dir, f"original.jpg"), result['original_image'])
        # 保存带特征点的图
        cv2.imwrite(os.path.join(output_dir, f"with_keypoints.jpg"), result['image_with_keypoints'])
        # 保存透明特征点图
        cv2.imwrite(os.path.join(output_dir, f"keypoints_only.png"), result['keypoints_only_image'])
        # 保存信息文件
        with open(os.path.join(output_dir, "ORBinfo.txt"), 'w') as f:
            f.write(f"参数配置:\n")
            f.write(f" ORB特征数: {self.num_orb_features}\n")
            f.write(f" 期望特征点数: {self.num_desired_keypoints}\n")
            f.write(f" 网格尺寸(行×列): {self.grid_size}\n")
            f.write(f"结果统计:\n")
            f.write(f" 初始特征点数: {len(result['keypoints'])}\n")
            f.write(f" 分布后特征点数: {result['num_keypoints']}\n")

        print(f"处理完成: {image_path} → 结果保存在 {output_dir}")

def main():
    """命令行接口，处理图像或目录。"""
    parser = argparse.ArgumentParser(description="ORB特征提取与均匀分布")
    parser.add_argument("input_path", help="输入图像路径或目录")
    parser.add_argument("-o", "--output_dir", default="results/ORBResult", help="输出目录")
    parser.add_argument("--num_orb", type=int, default=5000, help="ORB特征数量")
    parser.add_argument("--num_desired", type=int, default=500, help="期望特征点数")
    parser.add_argument("--grid_rows", type=int, default=50, help="网格行数")
    parser.add_argument("--grid_cols", type=int, default=50, help="网格列数")
    args = parser.parse_args()

    extractor = ORBFeatureExtractor(
        num_orb_features=args.num_orb,
        num_desired_keypoints=args.num_desired,
        grid_size=(args.grid_rows, args.grid_cols)
    )

    if os.path.isfile(args.input_path):
        extractor.process_and_save(args.input_path, args.output_dir)
    elif os.path.isdir(args.input_path):
        for fname in os.listdir(args.input_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(args.input_path, fname)
                extractor.process_and_save(img_path, args.output_dir)
    else:
        print("错误: 输入路径无效")

if __name__ == "__main__":
    main()
