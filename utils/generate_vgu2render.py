"""
收集从VGU到Render的Index映射
"""
import pandas as pd

render_df = pd.read_csv("./datasets/VGU_benchmark/annotations/T2I_Render.csv")
tgu_df = pd.read_csv("./datasets/VGU_benchmark/annotations/TGU.csv")

answers = tgu_df['answer'].tolist()
captions = render_df['caption'].tolist()

def get_duplicate_mapping(a, b):
    # 步骤1：构建「元素: 所有索引列表」的字典
    elem_indices = {}
    for idx, elem in enumerate(a):
        if elem not in elem_indices:
            elem_indices[elem] = []
        elem_indices[elem].append(idx)
    
    # 步骤2：按b的顺序提取索引序列（多对一映射）
    b_to_a_mapping = []
    for elem in b:
        b_to_a_mapping.append(elem_indices[elem])
    return b_to_a_mapping

def get_a2b_mapping(a, b):
    # a->b
    elem_indices = {}
    for idx, elem in enumerate(a):
        elem_indices[idx] = b.index(elem)
    return elem_indices

import json

# mapping = get_duplicate_mapping(answers, captions)
mapping = get_a2b_mapping(answers, captions)    # render->vgu
print(len(mapping))
with open("./datasets/VGU_benchmark/annotations/vgu2render_mapping.json", 'w') as f:
    json.dump(mapping, f)