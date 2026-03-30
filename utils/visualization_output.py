import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap
import os, json
import pandas as pd

# vgu2render
with open("./datasets/VGU_benchmark/annotations/vgu2render_mapping.json",'r') as f:
        vgu2render = json.load(f)

# label
tgu_csv_path = "./datasets/VGU_benchmark/annotations/TGU.csv"
# prediction
model_name = "LongCat"
sample_idx = 1339
work_dir  = "./workdirs"


df = pd.read_csv(tgu_csv_path)
tgu_path = os.path.join(work_dir,"TGU",model_name,f"{sample_idx}.txt")
vgu_path = os.path.join(work_dir,"VGU",model_name,f"{sample_idx}.png")
render_path = os.path.join(work_dir,"T2I_Render",model_name,f"{vgu2render[str(sample_idx)]}.png")

question = df.iloc[sample_idx, 1]
label_answer = df.iloc[sample_idx, 2]
data_source = df.iloc[sample_idx, -1]
if os.path.exists(tgu_path):
    with open(tgu_path, 'r') as f:
            tgu_prediciton = f.read()
else:
    tgu_prediciton = "None"
output_path = os.path.join("./codes/VGU/assets",f"{model_name}_{sample_idx}.jpg")


def make_board(top_texts, bottom_left_text, bottom_images, save_path="visualize.jpg", fig_size=(12, 6)):
    """
    2x3 布局：
    - 第一行：3 列放文本
    - 第二行：第一列放文本，第二、三列放图片
    """
    assert len(top_texts) == 3, "top_texts 需要包含3条文本"
    assert len(bottom_images) == 2, "bottom_images 需要包含2张图片"

    fig, axes = plt.subplots(2, 3, figsize=fig_size)
    # 第一行：文本
    for i in range(3):
        ax = axes[0, i]
        ax.axis('off')
        text = top_texts[i]
        wrapped = "\n".join(textwrap.wrap(text, width=40))
        ax.text(0.5, 0.5, wrapped, ha='center', va='center', fontsize=12, wrap=True)

    # 第二行：第一列文本，第二、三列图片
    # 第1列：文本
    ax = axes[1, 0]
    ax.axis('off')
    wrapped = "\n".join(textwrap.wrap(bottom_left_text, width=40))
    ax.text(0.5, 0.5, wrapped, ha='center', va='center', fontsize=12, wrap=True)

    # 第2列：图片1
    ax = axes[1, 1]
    ax.axis('off')
    p1 = bottom_images[0]
    if os.path.exists(p1):
        img1 = mpimg.imread(p1)
        ax.imshow(img1)
        ax.set_title("VGU")
    else:
        ax.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=12)

    # 第3列：图片2
    ax = axes[1, 2]
    ax.axis('off')
    p2 = bottom_images[1]
    if os.path.exists(p2):
        img2 = mpimg.imread(p2)
        ax.imshow(img2)
        ax.set_title("Render")
    else:
        ax.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# 示例用法
top_texts = [question, label_answer, data_source]
bottom_left_text = tgu_prediciton
bottom_images = [vgu_path, render_path]
print(f"Question:{question}\nAnswer:{label_answer}\nSource:{data_source}\nTGU:{tgu_prediciton}")
make_board(top_texts, bottom_left_text, bottom_images, save_path=output_path, fig_size=(12, 6))