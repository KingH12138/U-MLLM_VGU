import os, json, re
import pandas as pd


def read_json_tostr(json_path):
    try:
        with open(json_path, 'r') as f:
            judge_dict = json.load(f)
        return judge_dict
    except ValueError as e:
        raise e

def jsonstr_to_resultdict(jsonstr):
    """
    写如何从LLM输出的代码字符串中提取结果字典的逻辑
    """
    # 只解析score部分

    jsonstr = jsonstr.replace('json\n', "")
    jsonstr = jsonstr.replace('```', "")
    return eval(jsonstr)

def jsonstr_to_scores(jsonstr):
    # 1. 去 code fence
    jsonstr = re.sub(r"^```json\s*", "", jsonstr.strip())
    jsonstr = re.sub(r"\s*```$", "", jsonstr.strip())

    # 2. 保守截断 justification（只要前三个字段）
    m = re.search(
        r'\{\s*"correctness"\s*:\s*\d+\s*,\s*"completeness"\s*:\s*\d+\s*,\s*"legibility"\s*:\s*\d+',
        jsonstr,
        flags=re.S
    )
    if not m:
        raise ValueError("Score fields not found")

    head = m.group(0)

    # 3. 补齐 JSON
    safe_json = head + "\n}"

    return json.loads(safe_json)


def batch_proc0(model_name_list, task_name, save_dir):
    model_avg_result = {}
    for model_name in model_name_list:
        json_path = os.path.join(save_dir, task_name, f"{model_name}.json")
        judge_dict = read_json_tostr(json_path)
        num_sample = len(judge_dict)
        legibility = []
        completeness = []
        correctness = []
        for i in range(num_sample):
            try:
                eval_result = jsonstr_to_scores(judge_dict[i]['judge_result'][0])
            except:
                print(judge_dict[i]['judge_result'][0])
                raise ValueError("Stop.")
            legibility.append(eval_result['legibility'])
            completeness.append(eval_result['completeness'])
            correctness.append(eval_result['correctness'])
        model_avg_result[model_name] = {
            "legibility": sum(legibility)/num_sample,
            "completeness": sum(completeness)/num_sample,
            "correctness": sum(correctness)/num_sample
        }
    return model_avg_result

def batch_proc1(model_name_list, task_name, save_dir):
    """
    进行分类，找到completeness低下但correctness高的样本
    """
    model_avg_result = []
    for model_name in model_name_list:
        json_path = os.path.join(save_dir, task_name, f"{model_name}.json")
        judge_dict = read_json_tostr(json_path)
        num_sample = len(judge_dict)
        for i in range(num_sample):
            try:
                eval_result = jsonstr_to_scores(judge_dict[i]['judge_result'][0])
            except:
                print(judge_dict[i]['judge_result'][0])
                raise ValueError("Stop.")
            # 分类
            if eval_result['correctness']>4 and eval_result['completeness']>4:
                model_avg_result.append({
                    "legibility": eval_result['legibility'],
                    "completeness": eval_result['completeness'],
                    "correctness": eval_result['correctness'],
                    "prediction_path":judge_dict[i]['prediction_img_path'],
                })
    return model_avg_result

if __name__=="__main__":
    # 三大维度评价指标整理
    # single_proc()
    
    print("="*40)
    print("VGU:")
    save_dir = "./workdirs/COMSC_llm_judge"
    task_name = "VGU"
    MODELS_TO_EVALUATE = [
        "Emu3",
        "Bagel",
        "BLIP3o",
        "Janus",
        "JanusFlow", 
        "Show-o",
        "UniLIP",
        "LongCat",
        "Qwen-Image",
        "VGT"
    ]
    model_eval_result = batch_proc1(MODELS_TO_EVALUATE, task_name, save_dir)
    
    print(model_eval_result)
    #######################################################################
    print("="*40)
    print("Render:")
    save_dir = "./workdirs/COMSC_llm_judge"
    task_name = "Render"
    MODELS_TO_EVALUATE = [
        "Emu3",
        "Bagel",
        "BLIP3o",
        "Janus",
        "JanusFlow", 
        "Show-o",
        "UniLIP",
        "LongCat",
        "Qwen-Image",
        "VGT"
    ]
    model_eval_result = batch_proc1(MODELS_TO_EVALUATE, task_name, save_dir)
    print(model_eval_result)
    #######################################################################
    # 查看标注文件
    tgu_csv = pd.read_csv("./datasets/VGU_benchmark/annotations/TGU.csv")
    print(tgu_csv.iloc[1339,1])

