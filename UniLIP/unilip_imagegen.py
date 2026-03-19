from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
import torch
import sys
import os
from tqdm import tqdm
from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.mm_utils import get_model_name_from_path
from unilip.pipeline_gen import CustomGenPipeline
import random

def get_model():
    model_path = "/hongbojiang/checkpoints/kanashi6/UniLIP-3B"
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, multi_model, context_len = load_pretrained_model_general('UniLIP_InternVLForCausalLM', model_path, None, model_name)

    pipe = CustomGenPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer)

    return multi_model, pipe

def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""

    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image

def add_template(prompt):
    instruction = ('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n<img>')
    pos_prompt = instruction.format(input=prompt[0])

    cfg_prompt = instruction.format(input=prompt[1])
    return [pos_prompt, cfg_prompt]

def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_image(prompt, multi_model, pipe):
    generator = torch.Generator(device=multi_model.device).manual_seed(4)
    set_global_seed(seed=4)
    gen_img = pipe(add_template([f"Generate an image: {prompt}", "Generate an image."]), guidance_scale=3.0, generator=generator)
    return gen_img

if __name__=='__main__':
    prompt = 'What is the result of mixing red and blue paint?'
    model, pipe = get_model()
    image_sana = generate_image(prompt, model, pipe)
    save_path = f"/hongbojiang/codes/VGU/assets/unilip_{prompt}.jpg"
    image_sana.save(save_path)

