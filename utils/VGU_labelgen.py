import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
import math
from tqdm import tqdm

def generate_text_label(text, output_path, image_size=512, background_color='black', text_color='white', font_size=40, margin=20):
    """
    优化版：生成包含指定文本的标签图片
    改进：自动调整字体大小、合理的文本布局、垂直居中
    
    参数:
        text (str): 要渲染的文本内容
        output_path (str): 生成图片的保存路径
        image_size (int): 输出图片的尺寸（宽高相等）
        background_color (str): 背景颜色
        text_color (str): 文字颜色
        font_size (int): 初始字体大小（会自动调整）
        margin (int): 文字区域距离图片边缘的最小边距
    """
    
    # 1. 创建新图片
    image = Image.new('RGB', (image_size, image_size), color=background_color)
    draw = ImageDraw.Draw(image)
    
    # 2. 加载字体 - 修复字体加载逻辑
    font = load_best_font(font_size)
    
    # 3. 自动调整字体大小
    max_width = image_size - 2 * margin
    max_height = image_size - 2 * margin
    
    # 计算最佳字体大小
    optimal_font_size = calculate_optimal_font_size(
        text, font, max_width, max_height, font_size
    )
    
    # 重新加载优化后的字体
    if optimal_font_size != font_size:
        font = load_best_font(optimal_font_size)
    
    # 4. 智能文本换行
    lines = smart_text_wrap(text, font, max_width)
    
    # 5. 计算文本总高度和行高
    line_heights = []
    for line in lines:
        # 使用正确的文本边界框测量方法 [7,8](@ref)
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
    
    total_text_height = sum(line_heights)
    
    # 6. 垂直居中计算起始位置
    y_position = (image_size - total_text_height) // 2
    
    # 7. 逐行绘制（居中对齐，更美观）
    for i, line in enumerate(lines):
        line_height = line_heights[i]
        
        # 测量文本宽度
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        
        # 水平居中 [6,8](@ref)
        x_position = (image_size - text_width) // 2
        
        # 绘制文本
        draw.text((x_position, y_position), line, font=font, fill=text_color)
        
        y_position += line_height
    
    # 8. 保存图片
    image.save(output_path, 'PNG')
    return output_path

def load_best_font(font_size):
    """
    加载最佳可用字体
    """
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        'C:/Windows/Fonts/arial.ttf',  # Windows
        '/System/Library/Fonts/Arial.ttf',  # macOS
        'arial.ttf',
        'Arial.ttf'
    ]
    
    for path in font_paths:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        except (IOError, OSError):
            continue
    
    # 回退到默认字体
    print(f"警告：使用默认字体，大小: {font_size}")
    try:
        return ImageFont.load_default()
    except:
        # 如果默认字体也不可用，创建基本字体
        return ImageFont.load_default()

def calculate_optimal_font_size(text, font, max_width, max_height, initial_size):
    """
    计算最佳字体大小 [9,10,11](@ref)
    """
    # 创建临时draw对象进行测量
    temp_image = Image.new('RGB', (100, 100), color='black')
    temp_draw = ImageDraw.Draw(temp_image)
    
    best_size = initial_size
    
    # 从大到小测试字体大小
    for size in range(initial_size, 10, -1):
        try:
            test_font = load_best_font(size)
            
            # 测试换行
            lines = smart_text_wrap(text, test_font, max_width)
            
            # 计算总高度
            total_height = 0
            for line in lines:
                bbox = temp_draw.textbbox((0, 0), line, font=test_font)
                total_height += bbox[3] - bbox[1]
            
            # 检查是否适合可用空间
            if total_height <= max_height:
                best_size = size
                break
                
        except Exception as e:
            continue
    
    return best_size

def smart_text_wrap(text, font, max_width):
    """
    智能文本换行 [6](@ref)
    """
    # 如果文本为空，返回空列表
    if not text.strip():
        return [""]
    
    # 创建临时draw对象进行测量
    temp_image = Image.new('RGB', (100, 100), color='black')
    temp_draw = ImageDraw.Draw(temp_image)
    
    words = text.split()
    lines = []
    
    # 单个单词情况
    if len(words) == 0:
        return [text]
    
    current_line = []
    
    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        
        # 测量文本宽度 [8](@ref)
        bbox = temp_draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width > max_width:
            if len(current_line) > 1:
                # 移除最后一个词，开始新行
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # 单个词就超宽，完整保留但可能会超出边界
                lines.append(word)
                current_line = []
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

# 保留原有的批量处理函数（这些函数基本正确）
def batch_generate_labels(text_list, output_dir, index_list=None, **kwargs):
    """
    批量生成文本标签图片
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, text in tqdm(enumerate(text_list)):
        if index_list is not None and i < len(index_list):
            filename = f"{index_list[i]}.png"
        else:
            filename = f"{i}.png"
        output_path = os.path.join(output_dir, filename)
        generate_text_label(text, output_path, **kwargs)

def generate_text_label_tgucsv(tgu_path, answer_idx, index_idx, output_dir):
    """
    从CSV文件生成文本标签图片
    """
    df = pd.read_csv(tgu_path)
    
    if isinstance(index_idx, int):
        print(f"答案列: {df.columns[answer_idx]}, 索引列: {df.columns[index_idx]}")
    else:
        print(f"答案列: {df.columns[answer_idx]}")
    
    answers_list = df.iloc[:, answer_idx].astype(str).tolist()
    
    if isinstance(index_idx, int):
        index_list = df.iloc[:, index_idx].astype(str).tolist()
        batch_generate_labels(answers_list, output_dir, index_list)
    else:
        batch_generate_labels(answers_list, output_dir)
    
    return answers_list

if __name__ == "__main__":
    import pandas as pd
    
    answer_list = generate_text_label_tgucsv(
        "./datasets/VGU_benchmark/annotations/TGU.csv", 2, 0,
        "./datasets/VGU_benchmark/label_images/VGU"
    )

    answer_list = list(set(answer_list))

    # 生成短文本版本
    batch_generate_labels(
        answer_list, 
        "./datasets/VGU_benchmark/label_images/T2I_Render", 
        list(range(len(answer_list)))
    )

    # 分类文本长度
    types = []
    for answer in answer_list:
        if len(answer) <= 20:
            types.append("short")
        else:
            types.append("long")

    df = {"caption": answer_list, "type": types}
    df = pd.DataFrame(df)
    df.to_csv("./datasets/VGU_benchmark/annotations/T2I_Render.csv")