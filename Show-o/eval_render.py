############################################
import sys
sys.path.append("./codes/VGU/Show-o")
from showo_imagegen import get_model, generate_image
############################################
import os, random
import pandas as pd
from PIL import Image

def print_rank(strs):
    print(f"[Rank{os.environ['CUDA_VISIBLE_DEVICES']}]:{strs}")

def main():
    ############################################
    config, model, vq_model, uni_prompting, device = get_model()
    ############################################
    # TGU和VGU共用anno
    ############################################
    anno = pd.read_csv("./datasets/VGU_benchmark/annotations/T2I_Render.csv")
    ############################################
    vgu_prompt_path = './datasets/VGU_benchmark/prompts/T2I_Render/task_prompts'
    sample_num = len(anno)
    with open(vgu_prompt_path, 'r') as f:
        re_prompt = f.read().split("\n")
    # 一个字符串list
    prompts = random.choices(re_prompt, k=sample_num)    
    random.shuffle(prompts)
    # PIL.Image对象组成的list
    indexes = list(range(sample_num))
    captions = list(anno['caption'])
    ############################################
    output_dir = "./workdirs/T2I_Render/Show-o"
    ############################################
    os.makedirs(output_dir, exist_ok=True)

    for idx, prompt_template in enumerate(prompts):
        try:
            prompt = prompt_template.replace("{text}", f"{captions[idx]}")
            print_rank(f"[Rank{os.environ['CUDA_VISIBLE_DEVICES']}]\nPrompt:{prompt}")
            ############################################
            save_path = os.path.join(output_dir, f"{indexes[idx]}.png")
            if os.path.isfile(save_path):   # 如果存在就跳过
                print_rank(f"[{idx+1}/{sample_num}] Generated image has been already generated:{save_path}")
                continue
            image_sana = generate_image(prompt, config, model, vq_model, uni_prompting, device)
            image_sana.save(save_path)
            ############################################
            print_rank(f"[Rank{os.environ['CUDA_VISIBLE_DEVICES']}] [{idx+1}/{sample_num}]Saved: {save_path}")
        except ValueError as e:
            print_rank(f"[{idx+1}/{sample_num}] Index Wrong:{indexes[idx]}")
            print_rank(e)

if __name__ == "__main__":
    main()  

