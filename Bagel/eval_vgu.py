############################################
import sys
sys.path.append("/hongbojiang/codes/VGU/Bagel")
from bagel_imagegen import get_inferencer, set_config
############################################
import os, random
import pandas as pd
from PIL import Image

def print_rank(strs):
    print(f"[Rank{os.environ['CUDA_VISIBLE_DEVICES']}]:{strs}")

def main():
    ############################################
    inferencer = get_inferencer()
    inference_hyper = set_config()
    ############################################
    # TGU和VGU共用anno
    ############################################
    anno = pd.read_csv("/hongbojiang/datasets/VGU_benchmark/annotations/TGU.csv")
    ############################################
    vgu_prompt_path = '/hongbojiang/datasets/VGU_benchmark/prompts/VGU/task_prompts_regenerate'
    sample_num = len(anno)
    with open(vgu_prompt_path, 'r') as f:
        re_prompt = f.read().split("\n")
    # 一个字符串list
    prompts = random.choices(re_prompt, k=sample_num)    
    random.shuffle(prompts)
    # PIL.Image对象组成的list
    indexes = list(range(sample_num))
    questions = list(anno['question'])
    ############################################
    output_dir = "/hongbojiang/workdirs/VGU/Bagel"
    ############################################
    os.makedirs(output_dir, exist_ok=True)

    for idx, prompt_template in enumerate(prompts):
        try:
            prompt = prompt_template.replace("{question}", f"\n{questions[idx]}\n")
            print_rank(f"[{idx+1}/{sample_num}]\nPrompt:{prompt}")
            ############################################
            save_path = os.path.join(output_dir, f"{indexes[idx]}.png")
            if os.path.isfile(save_path):   # 如果存在就跳过
                print_rank(f"[{idx+1}/{sample_num}] Generated image has been already generated:{save_path}")
                continue
            output_dict, _ = inferencer(text=prompt, **inference_hyper)
            output_dict['image'].save(save_path)
            ############################################
            print_rank(f"[{idx+1}/{sample_num}]Saved: {save_path}")
        except ValueError as e:
            print_rank(f"[{idx+1}/{sample_num}] Index Wrong:{indexes[idx]}")
            print(e)

if __name__ == "__main__":
    main()  

