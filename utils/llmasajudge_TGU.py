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

def generate_answer(question:str, prediction:str, label:str, sys_prompt, user_prompt, model, processor):
    # replace <QUESTION>
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
                {"type": "text", "text": f"Text A:\n{prediction}"},
                {"type": "text", "text": f"Text B:\n{label}"},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text


def single_judge():
    model, processor = get_judge_llm()
    idx=2
    task_name = "TGU"
    model_name = "Bagel"
    prediction_output_dir = "/hongbojiang/workdirs"

    anno_dir = "/hongbojiang/datasets/VGU_benchmark/annotations"

    if task_name=="VGU":
        anno_path = os.path.join(anno_dir, "TGU.csv") 
    else:
        anno_path = os.path.join(anno_dir, f"{task_name}.csv") 
    task_output_dir = os.path.join(prediction_output_dir, task_name)
    anno_df = pd.read_csv(anno_path)

    prediction_path = os.path.join(task_output_dir, model_name, f'{idx}.txt')
    with open(prediction_path, 'r') as f:
        prediction = f.read()
    question = anno_df['question'][idx]
    answer = anno_df['answer'][idx]

    # system
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_system_prompt_tgu", 'r') as f:
        SYS_PROMPT = f.read()
    # user
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_user_prompt_tgu", 'r') as f:
        USER_PROMPT = f.read()

    print(f"Question:\n{question}\nAnswer:\n{answer}\nPrediction:\n{prediction}")

    judge_result = generate_answer(
        question, 
        prediction, answer,
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

    task_name = "TGU"
    prediction_output_dir = "/hongbojiang/workdirs"
    anno_dir = "/hongbojiang/datasets/VGU_benchmark/annotations"
    if task_name=="VGU":
        anno_path = os.path.join(anno_dir, "TGU.csv") 
    else:
        anno_path = os.path.join(anno_dir, f"{task_name}.csv") 
    task_output_dir = os.path.join(prediction_output_dir, task_name)
    anno_df = pd.read_csv(anno_path)
    num_sample = len(anno_df)

    # system
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_system_prompt_tgu", 'r') as f:
        SYS_PROMPT = f.read()
    # user
    with open("/hongbojiang/datasets/VGU_benchmark/LLM_judge/llm_judge_user_prompt_tgu", 'r') as f:
        USER_PROMPT = f.read()
    
    # 看一下有没有已经生成好的样本
    if os.path.exists(judge_json_path):
        with open(judge_json_path, 'r', encoding='utf-8') as f:
            save_dicts = json.load(f)
            print(f"Loading from existing judge output json:{judge_json_path}.")
    else:
        save_dicts = []
    start_idx = len(save_dicts)
    for idx in tqdm(range(start_idx, num_sample)):  # anno_df必须是从0开始才建议这样干
        # try:
        prediction_path = os.path.join(task_output_dir, model_name, f'{idx}.txt')
        with open(prediction_path, 'r') as f:
            prediction = f.read()
        question = anno_df['question'][idx]
        answer = anno_df['answer'][idx]
        judge_result = generate_answer(
            question,
            prediction, answer,
            SYS_PROMPT, USER_PROMPT, model, processor
        )
        save_dict = {
            "question":question,
            "answer":answer,
            "prediction":prediction,
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
                        required=True,
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