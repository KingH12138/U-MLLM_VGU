############################################
import sys
sys.path.append("/hongbojiang/codes/VGU/UniLIP")
from unilip_textgen import get_model, generate_text
############################################
import os, random
import pandas as pd
from PIL import Image

def print_rank(strs):
    print(f"[Rank{os.environ['CUDA_VISIBLE_DEVICES']}]:{strs}")

def main():
    ############################################
    model, tokenizer = get_model()
    ############################################
    # TGU和VGU共用anno
    ############################################
    anno = pd.read_csv("/hongbojiang/datasets/VGU_benchmark/annotations/TGU.csv")
    ############################################
    sample_num = len(anno)
    # PIL.Image对象组成的list
    indexes = list(range(sample_num))
    questions = list(anno['question'])
    ############################################
    output_dir = "/hongbojiang/workdirs/TGU/UniLIP"
    ############################################
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(sample_num):
        try:
            prompt = f"Give me the answer of the question first and then summarize it in a few sentences.\nQuestion:{questions[idx]}"
            print_rank(f"[{idx+1}/{sample_num}]\n Prompt:{prompt}")
            ############################################
            save_path = os.path.join(output_dir, f"{indexes[idx]}.txt")
            if os.path.isfile(save_path):   # 如果存在就跳过
                print_rank(f"[{idx+1}/{sample_num}] Generated answer has been already generated:{save_path}")
                continue
            prediction_answer = generate_text(prompt, model, tokenizer)
            with open(save_path, 'w') as f:
                f.write(prediction_answer)
            ############################################
            print_rank(f"[{idx+1}/{sample_num}] Saved: {save_path}")
        except ValueError as e:
            print_rank(f"[{idx+1}/{sample_num}] Index Wrong:{indexes[idx]}")
            print_rank(e)

if __name__ == "__main__":
    main()  

