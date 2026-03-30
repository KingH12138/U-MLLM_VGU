from diffusers import DiffusionPipeline
import torch


def get_model():
    model_name = "./checkpoints/Qwen/Qwen-Image"
    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    return pipe


def generate_image(prompt, pipe):
    negative_prompt = " " # Recommended if you don't use a negative prompt.

    # Generate with different aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["16:9"]

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width//3,
        height=height//3,
        num_inference_steps=30,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

    return image

if __name__ == "__main__":
    pipe = get_model()
    prompt = "Which animal is better at catching mice—cats or dogs? Please write the answer on the blackboard."
    image = generate_image(prompt, pipe)
    image.save(f"./codes/VGU/assets/qwenimage_{prompt}.jpg")