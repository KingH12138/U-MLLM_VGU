############################################
import sys
sys.path.append("/hongbojiang/codes/VGU/Janus")
from janusflow_imagegen import get_model, generate
############################################
import os, random
import pandas as pd
from PIL import Image

def print_rank(strs):
    print(f"[Rank{os.environ['CUDA_VISIBLE_DEVICES']}]:{strs}")

def main():
    ############################################
    vl_gpt, vae, vl_chat_processor = get_model()
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
    output_dir = "/hongbojiang/workdirs/VGU/JanusFlow"
    ############################################
    os.makedirs(output_dir, exist_ok=True)

    for idx, prompt_template in enumerate(prompts):
        try:
            prompt = prompt_template.replace("{question}", f"\n{questions[idx]}\n")
            print_rank(f"Prompt:{prompt}")
            ############################################
            save_path = os.path.join(output_dir, f"{indexes[idx]}.png")
            if os.path.isfile(save_path):   # 如果存在就跳过
                print_rank(f"[{idx+1}/{sample_num}] Generated image has been already generated:{save_path}")
                continue
            generate(
                vl_gpt, 
                vae, 
                vl_chat_processor, 
                prompt, save_path,
                cfg_weight=2.0, num_inference_steps=30, batchsize=1            
            )
            ############################################
            print_rank(f"[{idx+1}/{sample_num}]Saved: {save_path}")
        except ValueError as e:
            print_rank(f"[{idx+1}/{sample_num}] Index Wrong:{indexes[idx]}")
            print_rank(e)

if __name__ == "__main__":
    main()  

