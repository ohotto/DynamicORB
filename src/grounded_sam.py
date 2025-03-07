"""
python src/grounded_sam.py \
--config $(pwd)/configs/GroundingDINO_SwinT_OGC.py \
--grounded_checkpoint $(pwd)/models/groundingdino_swint_ogc.pth \
--sam_checkpoint $(pwd)/models/sam_vit_l_0b3195.pth \
--input_image $(pwd)/images/cars1.jpg \
--text_prompt "car . truck . motorbike . bicycle . people" \
--output_dir $(pwd)/results/main/

Grounded-SAM 实例分割程序使用说明：

该程序结合 GroundingDINO 和 SAM 模型，实现基于文本提示的图像实例分割

参数说明：
   - --config: GroundingDINO 配置文件路径（例如：GroundingDINO_SwinT_OGC.py）
   - --grounded_checkpoint: GroundingDINO 模型权重路径（例如：groundingdino_swint_ogc.pth）
   - --sam_checkpoint: SAM 模型权重路径（例如：sam_vit_l_0b3195.pth）【默认使用L尺寸，其他尺寸请自行修改代码__init__】
   - --input_image: 待处理的输入图像路径
   - --text_prompt: 检测提示文本，用 '.' 分隔多个类别（例如：'car . person . traffic light'）
   - --output_dir: 输出目录路径，默认为 "outputs"
   - --device: 计算设备选择，可以是 "cuda" 或 "cpu"，默认为 "cuda"（如果可用）

输出：
   - GSAoutput.jpg: 包含原始图像、检测框和分割掩膜的可视化结果
   - GSAmask.jpg: 分割掩膜的可视化图像，每个实例用不同的颜色表示
   - GSAmask.json: 包含每个实例的标签、置信度得分和边界框信息的 JSON 文件

注意事项：
   - 确保提供的模型权重文件路径正确
   - 根据实际需求调整文本提示，以获得最佳分割效果
   - 如果 GPU 可用，建议使用 "cuda" 设备以提高处理速度
"""


import os
import sys
import argparse
import numpy as np
import json
import torch
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) 

sys.path.insert(0, os.path.join(root_dir, "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.insert(0, os.path.join(root_dir, "Grounded-Segment-Anything", "segment_anything"))
sys.path.insert(0, os.path.join(root_dir, "Grounded-Segment-Anything")) 

from GroundingDINO.groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

class GroundedSAM:
    def __init__(self, config_path, grounded_checkpoint, sam_version="vit_l", sam_checkpoint=None, 
                 sam_hq_checkpoint=None, use_sam_hq=False, device="cpu", bert_path=None):
        """初始化Grounded-SAM模型"""
        self.device = device
        self.model = self._load_grounding_model(config_path, grounded_checkpoint, bert_path)
        self.sam_predictor = self._load_sam_predictor(sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq)
        
    def _load_grounding_model(self, config_path, checkpoint_path, bert_path):
        """加载Grounded-DINO检测模型"""
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        args.bert_base_uncased_path = bert_path  # 设置BERT模型路径
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model.to(self.device)
    
    def _load_sam_predictor(self, sam_version, sam_checkpoint, sam_hq_checkpoint, use_sam_hq):
        """加载SAM分割预测器"""
        if use_sam_hq:
            sam = sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint)
        else:
            sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        return SamPredictor(sam)
    
    def process_image(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """执行图像处理全流程"""
        # 加载并预处理图像
        image_pil, image_tensor = self._load_image(image_path)
        
        # 执行文本引导的检测
        boxes, phrases, logits = self._get_grounding_output(image_tensor, text_prompt, box_threshold, text_threshold)
        
        # 准备SAM分割
        image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_cv)
        
        # 调整检测框坐标格式
        boxes = self._adjust_boxes(boxes, image_pil.size)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, image_cv.shape[:2]).to(self.device)
        
        # 执行实例分割
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        return {
            "image_pil": image_pil,
            "original_size": image_pil.size,
            "masks": masks,
            "boxes": boxes,
            "phrases": phrases,
            "logits": logits,
            "image_cv": image_cv
        }
    
    def _load_image(self, image_path):
        """加载并预处理输入图像"""
        image_pil = Image.open(image_path).convert("RGB")
        transform = Compose([
            RandomResize([800], max_size=1333),  # 随机调整大小
            ToTensor(),                          # 转为张量
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image
    
    def _get_grounding_output(self, image_tensor, text_prompt, box_threshold, text_threshold):
        """执行文本引导的目标检测"""
        # 格式化提示文本（确保以句号结尾）
        caption = text_prompt.lower().strip() + "." if not text_prompt.endswith(".") else text_prompt
        image_tensor = image_tensor.to(self.device)
        
        # 执行检测推理
        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[caption])
        
        # 后处理检测结果
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        
        # 根据阈值过滤结果
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]
        logits_filt_scores = logits_filt.max(dim=1)[0].cpu().numpy()
        
        # 提取类别短语
        tokenized = self.model.tokenizer(caption)
        phrases = []
        for logit in logits_filt:
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, self.model.tokenizer))
        
        return boxes_filt, phrases, logits_filt_scores
    
    def _adjust_boxes(self, boxes, image_size):
        """调整检测框坐标格式"""
        H, W = image_size[1], image_size[0]
        # 将归一化坐标转换为像素坐标
        boxes = boxes * torch.tensor([W, H, W, H]).to(boxes.device)
        # 转换中心坐标格式为角点坐标格式
        boxes[:, :2] -= boxes[:, 2:] / 2  # 左上角 = 中心点 - 宽高/2
        boxes[:, 2:] += boxes[:, :2]       # 右下角 = 左上角 + 宽高
        return boxes.cpu()
    
    def visualize_results(self, results, output_path, show_phrases=True):
        """可视化分割结果"""
        plt.figure(figsize=(10, 10))
        plt.imshow(results["image_cv"])
        
        # 绘制所有掩膜
        for mask in results["masks"]:
            self._show_mask(mask.cpu().numpy(), plt.gca())
            
        # 绘制检测框和标签
        if show_phrases:
            for box, phrase, logit in zip(results["boxes"], results["phrases"], results["logits"]):
                self._show_box(box.numpy(), plt.gca(), f"{phrase} ({logit:.2f})")
        
        plt.axis('off')
        plt.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()
    
    def _show_mask(self, mask, ax, random_color=True):
        """在指定坐标轴上绘制掩膜（内部方法）"""
        color = np.concatenate([np.random.random(3), [0.6]]) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def _show_box(self, box, ax, label):
        """在指定坐标轴上绘制检测框（内部方法）"""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)
    
    def save_mask_data(self, output_dir, results):
        """保存掩膜元数据"""
        masks = results["masks"].cpu()
        boxes = results["boxes"].numpy()
        phrases = results["phrases"]
        logits = results["logits"]
        
        # 生成掩膜索引图
        mask_img = torch.zeros(masks.shape[-2:], dtype=torch.uint8)
        for idx, mask in enumerate(masks):
            mask_img[mask[0]] = idx + 1  # 背景为0，实例从1开始编号
        
        # 保存掩膜可视化
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'GSAmask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()
        
        # 生成元数据JSON
        json_data = [{'value': 0, 'label': 'background'}]
        for idx, (phrase, logit, box) in enumerate(zip(phrases, logits, boxes)):
            json_data.append({
                'value': idx + 1,
                'label': phrase,
                'logit': float(logit),
                'box': box.tolist()  # 转换numpy数组为列表
            })
        
        # 保存元数据文件
        with open(os.path.join(output_dir, 'GSAmask.json'), 'w') as f:
            json.dump(json_data, f, indent=4)

    def visualize_boxes_transparent(self, results, output_path):
        """在透明背景上绘制检测框和标签"""
        img_size = results['original_size']  # (width, height)
        transparent_img = Image.new('RGBA', img_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(transparent_img)
        
        for box, phrase, logit in zip(results['boxes'], results['phrases'], results['logits']):
            x0, y0, x1, y1 = box.cpu().numpy() if isinstance(box, torch.Tensor) else box
            # 绘制矩形框
            draw.rectangle([x0, y0, x1, y1], outline='green', width=2)
            # 绘制标签文本
            label = f"{phrase} ({logit:.2f})"
            draw.text((x0, y0), label, fill='black', stroke_width=1)
        
        transparent_img.save(output_path)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="Grounded-SAM实例分割命令行工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", 
                        required=True,
                        help="Grounded-DINO配置文件路径（例：GroundingDINO_SwinT_OGC.py）")
    parser.add_argument("--grounded_checkpoint", 
                        required=True,
                        help="Grounded-DINO模型权重路径（例：groundingdino_swint_ogc.pth）")
    parser.add_argument("--sam_checkpoint", 
                        required=True,
                        help="SAM模型权重路径（例：sam_vit_h_4b8939.pth）")
    parser.add_argument("--input_image", 
                        required=True,
                        help="待处理的输入图像路径")
    parser.add_argument("--text_prompt", 
                        required=True,
                        help="检测提示文本（用'.'分隔多个类别，例：'car . person . traffic light'）")
    parser.add_argument("--output_dir", 
                        default="outputs",
                        help="输出目录路径")
    parser.add_argument("--device", 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"],
                        help="计算设备选择")
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化处理器
    processor = GroundedSAM(
        config_path=args.config,
        grounded_checkpoint=args.grounded_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        device=args.device
    )
    
    # 执行处理流程
    results = processor.process_image(
        image_path=args.input_image,
        text_prompt=args.text_prompt
    )
    
    # 生成可视化结果
    processor.visualize_results(
        results=results,
        output_path=os.path.join(args.output_dir, "GSAoutput.jpg")
    )
    
    # 保存掩膜数据
    processor.save_mask_data(
        output_dir=args.output_dir,
        results=results
    )

if __name__ == "__main__":
    main()
