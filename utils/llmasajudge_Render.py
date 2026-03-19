from transformers import AutoModelForImageTextToText, AutoProcessor
import torch, os
import pandas as pd
from PIL import Image
import argparse
import time, json
from tqdm import tqdm

def print_rank(strs):
    print(f"[Rank{os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')}]:{strs}")

def get_judge_llm():
    # default: Load the model on the available device(s)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = AutoModelForImageTextToText.from_pretrained(
        "/hongbojiang/checkpoints/Qwen/Qwen2.5-VL-72B-Instruct",
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained("/hongbojiang/checkpoints/Qwen/Qwen2.5-VL-72B-Instruct")
    return model, processor

def generate_answer(question, prediction, label, sys_prompt, user_prompt, model, processor):
    # st = time.time()
    image1 = prediction
    image2 = label
    # replace <QUESTION> using question
    user_prompt = user_prompt.replace("<QUESTION>", question)
    messages = [
        {
            "role": "system", 
            "content": [
                {"type": "text", "text": sys_prompt}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image","image": image1},
                {"type": "image","image": image2},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    # print(messages)
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # ed = time.time()
    # print("Time consumed:{}s".format(ed-st))
    return output_text


def single_judge():
    model, processor = get_judge_llm()
    idx=2
    task_name = "T2I_Render"
    model_name = "Bagel"
    output_dir = "/hongbojiang/workdirs"
    label_image_dir = "/hongbojiang/datasets/VGU_benchmark/label_images"
    anno_dir = "/hongbojiang/datasets/VGU_benchmark/annotations"

    if task_name=="VGU":
        anno_path = os.path.join(anno_dir, "TGU.csv") 
    else:
        anno_path = os.path.join(anno_dir, f"{task_name}.csv") 
    task_label_dir = os.path.join(label_image_dir, task_name)
    task_output_dir = os.path.join(output_dir, task_name)
    anno_df = pd.read_csv(anno_path)

    label_img_path = os.path.join(task_label_dir,f'{idx}.png')
    prediction_img_path = os.path.join(task_output_dir, model_name, f'{idx}.png')
    caption = anno_df['caption'][idx]


    # system
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_system_prompt_render", 'r') as f:
        SYS_PROMPT = f.read()
    # user
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_user_prompt_render", 'r') as f:
        USER_PROMPT = f.read()

    judge_result = generate_answer(
        caption, 
        prediction_img_path, label_img_path,
        SYS_PROMPT, USER_PROMPT, model, processor
    )
    print(f"Judge_result:\n{judge_result}")

def batch_judge(args):
    model_name = args.model_name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  # 保存LLM judge结果目录
    judge_json_path = os.path.join(output_dir, f"{model_name}.json")
    save_steps=100

    model, processor = get_judge_llm()

    task_name = "T2I_Render"
    output_dir = "/hongbojiang/workdirs"
    label_image_dir = "/hongbojiang/datasets/VGU_benchmark/label_images"
    anno_dir = "/hongbojiang/datasets/VGU_benchmark/annotations"
    if task_name=="VGU":
        anno_path = os.path.join(anno_dir, "TGU.csv") 
    else:
        anno_path = os.path.join(anno_dir, f"{task_name}.csv") 
    task_label_dir = os.path.join(label_image_dir, task_name)
    task_output_dir = os.path.join(output_dir, task_name)
    anno_df = pd.read_csv(anno_path)
    num_sample = len(anno_df)

    # system
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_system_prompt_render", 'r') as f:
        SYS_PROMPT = f.read()
    # user
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_user_prompt_render", 'r') as f:
        USER_PROMPT = f.read()
    
    
    # 看一下有没有已经生成好的样本
    if os.path.exists(judge_json_path):
        with open(judge_json_path, 'r', encoding='utf-8') as f:
            save_dicts = json.load(f)
            print(f"Loading from existing judge output json:{judge_json_path}.")
    else:
        save_dicts = []
    start_idx = len(save_dicts)
    for idx in tqdm(range(start_idx, num_sample)):
        # try:
        label_img_path = os.path.join(task_label_dir,f'{idx}.png')
        prediction_img_path = os.path.join(task_output_dir, model_name, f'{idx}.png')
        caption = anno_df['caption'][idx]
        question = caption
        judge_result = generate_answer(
            question, 
            prediction_img_path, label_img_path,
            SYS_PROMPT, USER_PROMPT, model, processor
        )
        save_dict = {
            "question":question,
            "prediction_img_path":prediction_img_path,
            "label_img_path":label_img_path,
            "judge_result":judge_result
        }
        save_dicts.append(save_dict)
        if not (idx+1) % save_steps:
            with open(judge_json_path, 'w', encoding='utf-8') as f:
                json.dump(save_dicts, f, ensure_ascii=False, indent=4)
        # except:
        #     print_rank(f"Wrong:{prediction_img_path}.")
    
    with open(judge_json_path, 'w', encoding='utf-8') as f:
        json.dump(save_dicts, f, ensure_ascii=False, indent=4)
    
    print(f"Judge json has been saved to {judge_json_path}.")

def parse_args():
    parser = argparse.ArgumentParser(description='模型训练参数解析器')
    parser.add_argument('--output_dir',
                        type=str,
                        # required=True,
                        default="/hongbojiang/workdirs/COMSC_llm_judge",
                        help='json文件存放目录')
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='模型名词')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    batch_judge(args)
    # single_judge()