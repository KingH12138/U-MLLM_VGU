############################################
import sys
sys.path.append("./codes/VGU/Bagel")
from bagel_textgen import get_inferencer, set_config
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
    anno = pd.read_csv("./datasets/VGU_benchmark/annotations/TGU.csv")
    ############################################
    sample_num = len(anno)
    # PIL.Image对象组成的list
    indexes = list(range(sample_num))
    questions = list(anno['question'])
    ############################################
    output_dir = "./workdirs/TGU/Bagel"
    ############################################
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(sample_num):
        try:
            prompt = questions[idx]
            print_rank(f"[{idx+1}/{sample_num}]\n Prompt:{prompt}")
            ############################################
            save_path = os.path.join(output_dir, f"{indexes[idx]}.txt")
            if os.path.isfile(save_path):   # 如果存在就跳过
                print_rank(f"[{idx+1}/{sample_num}] Generated answer has been already generated:{save_path}")
                continue
            output_dict, _ = inferencer(text=prompt, understanding_output=True, **inference_hyper)
            prediction_answer = output_dict['text']
            with open(save_path, 'w') as f:
                f.write(prediction_answer)
            ############################################
            print_rank(f"[{idx+1}/{sample_num}] Saved: {save_path}")
        except ValueError as e:
            print_rank(f"[{idx+1}/{sample_num}] Index Wrong:{indexes[idx]}")
            print_rank(e)

if __name__ == "__main__":
    main()  

