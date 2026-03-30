import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
from xtuner.registry import BUILDER
from mmengine.config import Config
from einops import rearrange
import sys
sys.path.append("./codes/VGU/VGT")
from src.utils import load_checkpoint_with_ema

# Parameter settings
###############VGT Qwen2.5VL########################
CPKT_PATH = "./checkpoints/hustvl/vgt_qwen25vl_2B_sft/iter_5000.pth"
CONFIG = "./codes/VGU/VGT/configs/VGT_qwen2_5vl/vgt_qwen2_5vl_2B_448px.py"

###############VGT InterVL3########################
# CPKT_PATH = "ckpts/hustvl/vgt_internvl3_1_6B_sft/iter_5000.pth"
# CONFIG = "configs/VGT_internvl3/vgt_internvl3_1_6B_448px.py"

# default
CFG_SCALE = 4.5
NUM_STEPS = 30
HEIGHT = 448
WIDTH = 448
GRID_SIZE = 1
# acc_ratios = [1,2,8,16] # 1(next token), 4, 16, 32, ..., 256
MODEL_NAME = os.path.splitext(os.path.basename(CONFIG))[0]


def _generate_text_to_image(model, prompts, cfg_scale=3.5, num_steps=50,
                           height=512, width=512, temperature=1.0, grid_size=2, **kwargs):
    """
    Text-to-image generation task (prompts can be a single string or a list)
    
    Args:
        return_list: bool, whether to return a list of images
    
    Returns:
        PIL.Image or list[PIL.Image]
    """
    if not isinstance(prompts, list):
        prompts = [prompts]

    bsz = grid_size ** 2
    
    # Use new batch text condition preparation function
    batch_text_conditions = model.prepare_batch_text_conditions(prompts)
    input_ids = batch_text_conditions['input_ids']
    attention_mask = batch_text_conditions['attention_mask']
    
    # input_ids and attention_mask already contain the mixture of prompt and CFG
    # First half is prompt, second half is CFG
    total_prompts = len(prompts)
    seq_len = input_ids.shape[1]
    
    # Reshape to [total_prompts, 2, seq_len] format, where 2 represents prompt+CFG
    assert 2*total_prompts == input_ids.shape[0], "prepare_batch_text_conditions will return input_ids containing cfg. [p1,p2,p3,cfg1,cfg2,cfg3]"
    
    # Expand batches
    prompt_input_ids = input_ids[:total_prompts].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    prompt_attention_mask = attention_mask[:total_prompts].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    cfg_input_ids = input_ids[total_prompts:].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    cfg_attention_mask = attention_mask[total_prompts:].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    
    # Concatenate into [prompt1, prompt2, ..., cfg1, cfg2, ...] format
    batch_input_ids = torch.cat([prompt_input_ids, cfg_input_ids], dim=0)
    batch_attention_mask = torch.cat([prompt_attention_mask, cfg_attention_mask], dim=0)

    if cfg_scale == 1.0:
        batch_input_ids = batch_input_ids[:bsz]
        batch_attention_mask = batch_attention_mask[:bsz]

    # Generate images
    samples = model.generate(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        cfg_scale=cfg_scale,
        num_steps=num_steps,
        height=height,
        width=width,
        temperature=temperature,
        num_image_pre_caption=bsz,
        **kwargs
    )
    
    gen_images = []
    # Split results and process images corresponding to each prompt
    for i, prompt in enumerate(prompts):
        # Each prompt corresponds to bsz images, directly take corresponding positions
        start_idx = i * bsz
        end_idx = start_idx + bsz
        prompt_samples = samples[start_idx:end_idx]
        
        prompt_samples = rearrange(prompt_samples, '(m n) c h w -> (m h) (n w) c', m=grid_size, n=grid_size)
        prompt_samples = torch.clamp(
            127.5 * prompt_samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        gen_images.append(Image.fromarray(prompt_samples))

    
    return gen_images



def load_model(CONFIG_PATH, model_path):
    """Load model and weights"""
    # Load configuration and model
    config = Config.fromfile(CONFIG_PATH)
    model = BUILDER.build(config.model).cuda().bfloat16().eval()

    print("Model loading completed!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model precision: {next(model.parameters()).dtype}")

    # Load checkpoint
    if model_path is not None:
        load_checkpoint_with_ema(model, model_path, use_ema=True, map_location='cpu', strict=False)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        print(f"Weights loading completed: {model_path}")
    else:
        print("No checkpoint path specified")

    return model

def get_model():
    model = load_model(CONFIG, model_path=CPKT_PATH) 
    return model

def generate_image(prompt, model):
    prompts = [prompt]
    # ======== Generate images ========
    samples = _generate_text_to_image(
        model=model,
        prompts=prompts,
        cfg_scale=CFG_SCALE,
        num_steps=NUM_STEPS,
        height=HEIGHT,
        width=WIDTH,
        seed=42,
        grid_size=GRID_SIZE,
        scheduler_type="random",
        acc_ratio=1,
    )

    # ======== Save images ========
    sample = samples[0]
    # Convert to PIL Image
    if isinstance(sample, np.ndarray):
        if sample.max() <= 1.0:
            sample = (sample * 255).astype(np.uint8)
        else:
            sample = sample.astype(np.uint8)
        img = Image.fromarray(sample)
    else:
        img = sample
    return img