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

def main():
    parser = argparse.ArgumentParser(description="调整图片尺寸并保存到同目录下。", add_help=False)  # Disable default -h
    parser.add_argument("img_path", type=str, help="图片路径")
    parser.add_argument("-w", "--width", type=int, help="目标宽度 (像素)", required=False)
    parser.add_argument("-H", "--height", type=int, help="目标高度 (像素)", required=False)  # Now -H is for height
    parser.add_argument("-m", "--max_size", type=int, help="最长边像素数", required=False)
    parser.add_argument("--help", action='help', help='显示帮助信息')  # Add a --help option

    args = parser.parse_args()

    try:
        # 打开图片
        img = Image.open(args.img_path)
        # 调整图片尺寸
        resized_img = resize_image(img, width=args.width, height=args.height, max_size=args.max_size)
        
        # 保存图片到同目录下，文件名添加 "_resized" 后缀
        file_name, file_ext = os.path.splitext(args.img_path)
        output_path = file_name + "_resized" + file_ext
        resized_img.save(output_path)
        print(f"图片已调整尺寸并保存到: {output_path}")

    except FileNotFoundError:
        print(f"错误: 文件未找到: {args.img_path}")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
