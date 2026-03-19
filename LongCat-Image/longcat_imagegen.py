import torch
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline

def get_model():
    device = torch.device('cuda')
    checkpoint_dir = '/hongbojiang/checkpoints/meituan-longcat/LongCat-Image'

    text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
    transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
        torch_dtype=torch.bfloat16, use_safetensors=True).to(device)

    pipe = LongCatImagePipeline.from_pretrained(
        checkpoint_dir,
        transformer=transformer,
        text_processor=text_processor,
        torch_dtype=torch.bfloat16
    )
    # pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
    pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~17 GB); slower but prevents OOM

    return pipe

def generate_image(prompt, pipe):
    image = pipe(
        prompt,
        height=384,
        width=672,
        guidance_scale=4.5,
        num_inference_steps=30,
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(43),
        enable_cfg_renorm=True,
        enable_prompt_rewrite=True  # Reusing the text encoder as a built-in prompt rewriter
    ).images[0]
    return image

if __name__ == "__main__":
    prompt = "Which animal is better at catching mice—cats or dogs? "
    pipe = get_model()
    image = generate_image(prompt, pipe)
    image.save(f"/hongbojiang/codes/VGU/assets/longcat_{prompt}.jpg")