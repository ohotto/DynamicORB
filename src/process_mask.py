"""
python src/process_mask.py \
  --config $(pwd)/configs/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint $(pwd)/models/groundingdino_swint_ogc.pth \
  --sam_checkpoint $(pwd)/models/sam_vit_l_0b3195.pth \
  --input_image $(pwd)/images/cars1.jpg \
  --text_prompt "car . truck . motorbike . bicycle . people" \
  --output_dir $(pwd)/test/ \
  --dilation_kernel 5 \
  --dilation_iters 1
"""

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from grounded_sam import GroundedSAM

class MaskProcessor:
    def __init__(self, config_path, grounded_checkpoint, sam_checkpoint, device="cuda"):
        self.gsam = GroundedSAM(
            config_path=config_path,
            grounded_checkpoint=grounded_checkpoint,
            sam_checkpoint=sam_checkpoint,
            device=device
        )

    def _save_transparent_mask(self, mask_array, output_path, color=(0, 255, 255, 180)):
        """保存透明背景的分割mask"""
        rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        mask = (mask_array > 0)  # 处理任意非零值为mask
        rgba[mask] = color
        rgba[~mask] = [0, 0, 0, 0]
        Image.fromarray(rgba).save(output_path)
        
    def process(self, image_path, output_dir, 
                text_prompt, 
                box_threshold=0.3, 
                text_threshold=0.25,
                dilation_kernel_size=15,
                dilation_iterations=3):
        """处理单张图片的全流程"""
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行基础分割
        results = self.gsam.process_image(
            image_path=image_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # 保存中间结果
        self._save_intermediate_results(results, output_dir)
        
        # 合并所有mask
        merged_mask = self._merge_masks(results["masks"])
        seg_transparent_path = os.path.join(output_dir, "segmentation_mask_transparent.png")
        self._save_transparent_mask(merged_mask, seg_transparent_path)
        
        # 形态学膨胀
        dilated_mask = self._dilate_mask(merged_mask, dilation_kernel_size, dilation_iterations)
        dilated_transparent_path = os.path.join(output_dir, "dilated_mask_transparent.png")
        self._save_transparent_mask(dilated_mask, dilated_transparent_path)

        
        # 保存最终mask
        final_mask_path = os.path.join(output_dir, "processed_mask.jpg")
        self._save_mask_image(dilated_mask, final_mask_path)
        
        # 计算面积比例
        area_ratio = self._calculate_area_ratio(dilated_mask)
        
        # 保存统计信息
        self._save_metadata(output_dir, area_ratio, dilation_kernel_size, dilation_iterations)
        
        return {
            "merged_mask": merged_mask,
            "dilated_mask": dilated_mask,
            "area_ratio": area_ratio,
            "final_mask_path": final_mask_path
        }
    
    def _save_intermediate_results(self, results, output_dir):
        """保存原始分割结果"""
        # 原有可视化结果
        self.gsam.visualize_results(
            results=results,
            output_path=os.path.join(output_dir, "GSAoutput.jpg")
        )
        # 新增透明检测框
        self.gsam.visualize_boxes_transparent(
            results=results,
            output_path=os.path.join(output_dir, "detection_boxes_transparent.png")
        )
        # 保存mask数据
        self.gsam.save_mask_data(
            output_dir=output_dir,
            results=results
        )
    
    def _merge_masks(self, masks_tensor):
        """合并所有目标mask"""
        # masks_tensor形状: (N, 1, H, W)
        merged = torch.any(masks_tensor.squeeze(1), dim=0).cpu().numpy()
        return merged.astype(np.uint8) * 255
    
    def _dilate_mask(self, mask, kernel_size, iterations):
        """形态学膨胀处理"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        return dilated
    
    def _save_mask_image(self, mask_array, output_path):
        """保存mask为图像文件"""
        Image.fromarray(mask_array).save(output_path)
    
    def _calculate_area_ratio(self, mask_array):
        """计算mask面积占比"""
        total_pixels = mask_array.shape[0] * mask_array.shape[1]
        masked_pixels = np.sum(mask_array > 127)  # 考虑膨胀后的灰度值
        return masked_pixels / total_pixels
    
    def _save_metadata(self, output_dir, ratio, kernel_size, iterations):
        """保存处理参数和统计信息"""
        with open(os.path.join(output_dir, "PMASKinfo.txt"), "w") as f:
            f.write(f"Mask Area Ratio: {ratio:.4f}\n")
            f.write(f"Dilation Kernel Size: {kernel_size}\n")
            f.write(f"Dilation Iterations: {iterations}\n")

def main():
    parser = argparse.ArgumentParser(description="Process segmentation masks")
    parser.add_argument("--config", required=True, help="Grounded-SAM配置文件路径")
    parser.add_argument("--grounded_checkpoint", required=True, help="Grounded检测器权重路径")
    parser.add_argument("--sam_checkpoint", required=True, help="SAM模型权重路径")
    parser.add_argument("--input_image", required=True, help="输入图像路径")
    parser.add_argument("--text_prompt", required=True, help="分割提示文本")
    parser.add_argument("--output_dir", default="processed_output", help="输出目录")
    parser.add_argument("--box_thresh", type=float, default=0.3, help="检测框置信度阈值")
    parser.add_argument("--text_thresh", type=float, default=0.25, help="文本匹配阈值")
    parser.add_argument("--dilation_kernel", type=int, default=15, help="膨胀核大小")
    parser.add_argument("--dilation_iters", type=int, default=3, help="膨胀迭代次数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    
    args = parser.parse_args()
    
    processor = MaskProcessor(
        config_path=args.config,
        grounded_checkpoint=args.grounded_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        device=args.device
    )
    
    processor.process(
        image_path=args.input_image,
        output_dir=args.output_dir,
        text_prompt=args.text_prompt,
        box_threshold=args.box_thresh,
        text_threshold=args.text_thresh,
        dilation_kernel_size=args.dilation_kernel,
        dilation_iterations=args.dilation_iters
    )

if __name__ == "__main__":
    main()
