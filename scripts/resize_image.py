"""
图片尺寸调整工具。

可以调整单个图片或目录下所有图片的尺寸，并保存到指定目录或原目录。

使用方法:

1.  调整单个图片尺寸:
    python scripts/resize_image.py <图片路径> [-w <宽度>] [-H <高度>] [-m <最长边>] [--output <输出目录>]

    例如:
    python scripts/resize_image.py image.jpg -w 800  # 调整图片宽度为 800 像素，高度按比例缩放
    python scripts/resize_image.py image.jpg -H 600  # 调整图片高度为 600 像素，宽度按比例缩放
    python scripts/resize_image.py image.jpg -m 1024 # 调整图片最长边为 1024 像素
    python scripts/resize_image.py image.jpg -w 800 --output resized_images  # 调整宽度并保存到 resized_images 目录

2.  调整目录下所有图片尺寸:
    python scripts/resize_image.py <目录路径> [-w <宽度>] [-H <高度>] [-m <最长边>] [--output <输出目录>]

    例如:
    python scripts/resize_image.py images/ -w 800  # 调整 images 目录下所有图片宽度为 800 像素
    python scripts/resize_image.py images/ -m 512 --output resized_images  # 调整 images 目录下所有图片最长边为 512 像素并保存到 resized_images 目录

参数说明:
    <图片路径>: 要调整尺寸的图片文件路径。
    <目录路径>: 包含要调整尺寸的图片的目录路径。
    -w, --width: 目标宽度 (像素)。
    -H, --height: 目标高度 (像素)。
    -m, --max_size: 最长边像素数。
    --output: 输出目录。如果未指定，则将调整后的图片保存到与原始图片相同的目录，文件名添加 "_resized" 后缀。
    --help: 显示帮助信息。

注意:
    必须指定 width, height, 或 max_size 中的至少一个。
    不能同时指定 max_size 和 width 或 height。
"""
from PIL import Image
import os
import argparse

def resize_image(img, width=None, height=None, max_size=None):
    """
    调整图片尺寸。

    Args:
        img (PIL.Image): 图片对象。
        width (int, optional): 目标宽度，像素。 Defaults to None.
        height (int, optional): 目标高度，像素。 Defaults to None.
        max_size (int, optional): 最长边像素数。 Defaults to None.

    Returns:
        PIL.Image: 调整后的图片对象。
    """

    if not any([width, height, max_size]):
        raise ValueError("必须指定 width, height, 或 max_size 中的至少一个。")

    if max_size and (width or height):
        raise ValueError("不能同时指定 max_size 和 width 或 height。")

    original_width, original_height = img.size

    if width and height:
        # 同时指定宽度和高度，直接缩放到指定尺寸 (可能改变宽高比)
        img = img.resize((width, height), Image.LANCZOS)
    elif width:
        # 根据宽度计算高度比例
        ratio = width / float(original_width)
        new_height = int(original_height * ratio)
        img = img.resize((width, new_height), Image.LANCZOS)
    elif height:
        # 根据高度计算宽度比例
        ratio = height / float(original_height)
        new_width = int(original_width * ratio)
        img = img.resize((new_width, height), Image.LANCZOS)
    elif max_size:
        # 调整最长边
        if original_width > original_height:
            ratio = max_size / float(original_width)
            new_width = max_size
            new_height = int(original_height * ratio)
        else:
            ratio = max_size / float(original_height)
            new_height = max_size
            new_width = int(original_width * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    return img

def process_image(img_path, width=None, height=None, max_size=None, output_dir=None):
    """处理单个图片"""
    try:
        # 打开图片
        img = Image.open(img_path)
        # 调整图片尺寸
        resized_img = resize_image(img, width=width, height=height, max_size=max_size)
        
        # 确定输出路径
        file_name, file_ext = os.path.splitext(img_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
            output_path = os.path.join(output_dir, os.path.basename(file_name) + "_resized" + file_ext)
        else:
            output_path = file_name + "_resized" + file_ext
        
        # 保存图片
        resized_img.save(output_path)
        print(f"图片已调整尺寸并保存到: {output_path}")

    except FileNotFoundError:
        print(f"错误: 文件未找到: {img_path}")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="调整图片尺寸并保存。", add_help=False)
    parser.add_argument("path", type=str, help="图片路径或目录路径")
    parser.add_argument("-w", "--width", type=int, help="目标宽度 (像素)", required=False)
    parser.add_argument("-H", "--height", type=int, help="目标高度 (像素)", required=False)
    parser.add_argument("-m", "--max_size", type=int, help="最长边像素数", required=False)
    parser.add_argument("--output", type=str, help="输出目录", required=False)
    parser.add_argument("--help", action='help', help='显示帮助信息')

    args = parser.parse_args()

    if os.path.isfile(args.path):
        # 处理单个图片
        process_image(args.path, width=args.width, height=args.height, max_size=args.max_size, output_dir=args.output)
    elif os.path.isdir(args.path):
        # 处理目录下的所有图片
        for filename in os.listdir(args.path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(args.path, filename)
                process_image(img_path, width=args.width, height=args.height, max_size=args.max_size, output_dir=args.output)
    else:
        print(f"错误: 提供的路径既不是文件也不是目录: {args.path}")

if __name__ == "__main__":
    main()
