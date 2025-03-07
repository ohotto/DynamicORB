"""
该脚本使用 Grounding DINO 对一批图像执行目标检测

参数：
--config_file: Grounding DINO 模型配置文件路径
--checkpoint_path: Grounding DINO 模型权重文件路径
--image_path: 图像目录路径
--text_prompt: 检测提示文本
--output_dir: 输出目录路径
--box_threshold: 检测框置信度阈值，默认为 0.3
--text_threshold: 文本匹配阈值，默认为 0.25
--token_spans: 指定检测提示文本的 token 范围，格式为 [(start1, end1), (start2, end2), ...]
--cpu-only: 仅使用 CPU 运行
--extensions: 图像文件扩展名列表，默认为 ["jpg", "png", "jpeg", "bmp"]

输出：
输出目录中包含处理后的图像，文件名为原始图像文件名加上 "_pred.jpg" 后缀

示例：
python src/batch_grounding_dino.py \
    --config_file configs/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path models/groundingdino_swint_ogc.pth \
    --image_path images/ \
    --text_prompt "a cat . a dog ." \
    --output_dir results/
"""
import argparse
import os
import sys
from glob import glob
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) 

sys.path.insert(0, os.path.join(root_dir, "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.insert(0, os.path.join(root_dir, "Grounded-Segment-Anything")) 
sys.path.append(os.path.join(current_dir, '..'))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.vl_utils import create_positive_map_from_span
from scripts.resize_image import resize_image

def process_single_image(image_path, image_pil, output_dir, model, args):
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Run model inference
    image, _ = load_image(image_pil)
    
    # Run model inference
    boxes_filt, pred_phrases = get_grounding_output(
        model, 
        image, 
        args.text_prompt, 
        args.box_threshold, 
        text_threshold=args.text_threshold,
        cpu_only=args.cpu_only,
        token_spans=eval(f"{args.token_spans}") if args.token_spans else None
    )
    
    # Visualize and save predictions
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    pred_image_path = os.path.join(output_dir, f"{base_name}_pred.jpg")
    image_with_box.save(pred_image_path)


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_pil):
    # load image
    # image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image, image_pil


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO Batch Processing", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--image_path", "-i", type=str, required=True,
                        help="Path to image directory")
    parser.add_argument("--text_prompt", "-t", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--token_spans", type=str, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--extensions", nargs="+", default=["jpg", "png", "jpeg", "bmp"],
                        help="Supported image extensions")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    model = load_model(args.config_file, args.checkpoint_path, args.cpu_only)

    # 获取所有图片文件
    image_files = []
    for ext in args.extensions:
        image_files += glob(os.path.join(args.image_path, f"*.{ext}"))
        image_files += glob(os.path.join(args.image_path, f"*.{ext.upper()}"))

    if not image_files:
        print(f"No images found in {args.image_path} with extensions {args.extensions}")
        sys.exit(1)

    # 处理每张图片
    for img_path in image_files:
        print(f"Processing {img_path}...")
        try:
            image_pil = Image.open(img_path).convert("RGB")
            # 调整图片大小，可注释
            image_pil = resize_image(image_pil, height=480)
            process_single_image(img_path, image_pil, args.output_dir, model, args)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    print(f"Processing completed. Results saved to {args.output_dir}")
