import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

"""
UniLIP只对InternVL3中的ViT部分进行了Retrain
"""

def get_model():
    path = 'OpenGVLab/InternVL3-2B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    return model, tokenizer

def generate_text(question, model, tokenizer):
    # pure-text conversation (纯文本对话)
    generation_config = dict(
        max_new_tokens=1024,  # 提升最大生成长度，避免截断
        do_sample=False,  # 关闭采样（采样会降低指令依从性，优先确定性生成）
        temperature=0.01,  # 温度趋近0，强制模型严格遵循指令（越小越确定）
        top_p=0.0,  # 关闭top_p采样，进一步增强确定性
        repetition_penalty=1.1,  # 轻微重复惩罚，避免废话
        num_beams=4,  # 束搜索（提升生成质量和指令依从性）
        early_stopping=True,  # 束搜索完成后停止，避免无意义生成
        eos_token_id=tokenizer.eos_token_id,  # 明确结束符
        pad_token_id=tokenizer.pad_token_id
    )
    response, _ = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    return response


if __name__ == "__main__":
    question = 'What is the result of mixing red and blue paint?'
    model, tokenizer = get_model()
    answer = generate_text(question, model, tokenizer)
    print(answer)