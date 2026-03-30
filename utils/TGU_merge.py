"""
将TGU格式数据集随机采样并合并
"""
from datasets import load_dataset
import random
import pandas as pd
############################################################
# MMLU
MMLU_ds = load_dataset("./datasets/TextOnlyQA/cais/mmlu", "all")
# test: ['question', 'subject', 'choices', 'answer']
MMLU_ds_sampled = random.sample(list(MMLU_ds['test']), 250)

# BoolQ
BoolQ_ds = load_dataset("./datasets/TextOnlyQA/google/boolq")
# validation:['question', 'answer', 'passage']
BoolQ_ds_sampled = random.sample(list(BoolQ_ds['validation']), 250)

# CommmonSenseQA
CSQA_ds = load_dataset("./datasets/TextOnlyQA/tau/commonsense_qa")
# validation:['id', 'question', 'question_concept', 'choices', 'answerKey']
CSQA_ds_sampled = random.sample(list(CSQA_ds['validation']), 250)
############################################################
# ARC
ARC_ds = load_dataset("./datasets/TextOnlyQA/allenai/ai2_arc", "ARC-Challenge")
# validation:['id', 'question', 'choices', 'answerKey']
ARC_ds_sampled = random.sample(list(ARC_ds['validation']), 250)

# HellaSwag
hellaswag_ds = load_dataset("./datasets/TextOnlyQA/Rowan/hellaswag")
# validation:['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label']
hellaswag_ds_sampled = random.sample(list(hellaswag_ds['validation']), 250)

# OpenBookQA
openbook_ds = load_dataset("./datasets/TextOnlyQA/allenai/openbookqa")
# ['id', 'question_stem', 'choices', 'answerKey']
openbook_ds_sampled = random.sample(list(openbook_ds['validation']), 250)
############################################################
# MATH
math_ds = load_dataset("./datasets/TextOnlyQA/HuggingFaceH4/MATH-500")
# test: ['problem', 'solution', 'answer', 'subject', 'level', 'unique_id']
math_ds_sampled = random.sample(list(math_ds['test']), 250)

# HumanEval
humaneval_ds = load_dataset("./datasets/TextOnlyQA/openai/openai_humaneval")
# test:['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point']
if len(humaneval_ds['test'])<=250:
    humaneval_ds_sampled = humaneval_ds['test']
else:
    humaneval_ds_sampled = random.sample(list(humaneval_ds['test']), 250)

# GPQA
gpqa_ds = load_dataset("./datasets/TextOnlyQA/Wanfq/gpqa","gpqa_main")
# train(其实是val):["Question", "Correct Answer"]
gpqa_ds_sampled = random.sample(list(gpqa_ds['train']), 250)
############################################################

def format_choices2str(choices_strlist):
    output = []
    for i,choice in enumerate(choices_strlist):
        letter = chr(65+i)
        format_line = f"({letter}){choice}"
        output.append(format_line)
    return '\n'.join(output)
"""
经过每个采样后，我们需要给每个采集到样本进行问题重组、样本检查、合并
"""
col_names = ['question', 'answer', 'source']
VGU_dataset = []

# MMLU
for item in MMLU_ds_sampled:
    q = item['question']
    cs = item['choices']
    cs_string = format_choices2str(cs)
    re_question = f"{q} Please select the letter corresponding to the answer you believe to be correct from the options below.\n{cs_string}"
    re_answer = chr(65+item['answer'])  # 选项字母
    data_src = "MMLU"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# BoolQ
for item in BoolQ_ds_sampled:
    question = item['question']
    passage = item['passage']
    answer = item['answer']
    re_question = f"Read the passage and answer my question with True or False.\nPassage:{passage}\n Question:{question}"
    re_answer = str(item['answer'])
    data_src = "BoolQ"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# CommonSenseQA
for item in CSQA_ds_sampled:
    question = item['question']
    choices = [f"({chr(65+i)}){item['choices']['text'][i]}" for i in range(len(item['choices']['label']))]
    cs_string = '\n'.join(choices)
    re_question = f"{question} Please select the letter corresponding to the answer you believe to be correct from the options below.\n{cs_string}"
    re_answer = str(item['answerKey'])
    data_src = "CommonSenseQA"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# ARC
for item in ARC_ds_sampled:
    question = item['question']
    choices = [f"({chr(65+i)}){item['choices']['text'][i]}" for i in range(len(item['choices']['label']))]
    cs_string = '\n'.join(choices)
    re_question = f"{question} Please select the letter corresponding to the answer you believe to be correct from the options below.\n{cs_string}"
    re_answer = str(item['answerKey'])
    data_src = "ARC-Challenge"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# HellaSwag
for item in hellaswag_ds_sampled:
    scenario = item['ctx']
    cs = item['endings']
    cs_string = format_choices2str(cs)
    re_question = f"""Based on the given scenario, please select the most plausible continuation of the story.
scenario:
{scenario}
Choices:
{cs_string}
    """
    re_answer = chr(65+int(item['label']))  # 选项字母
    data_src = "HellaSwag"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )  
# OpenBookQA
for item in openbook_ds_sampled:
    question = item['question_stem']
    choices = [f"({chr(65+i)}){item['choices']['text'][i]}" for i in range(len(item['choices']['label']))]
    cs_string = '\n'.join(choices)
    re_question = f"{question} Please select the letter corresponding to the answer you believe to be correct from the options below.\n{cs_string}"
    re_answer = str(item['answerKey'])
    data_src = "OpenBookQA"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# MATH
for item in math_ds_sampled:
    question = item['problem']
    solution = item['solution']
    answer = item['answer']
    re_question = f"""Answer the question and provide the solution process.
Question:{question}
    """
    re_answer = f"""My solution:{solution}
Therefore, my answer is '{answer}'.
    """
    data_src = "MATH"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# HumanEval
for item in humaneval_ds_sampled:
    prompt = item['prompt']
    re_question = f"Complete the following Python function:\n{prompt}"
    re_answer = item['canonical_solution']
    data_src = "HumanEval"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# GPQA
for item in gpqa_ds_sampled:
    question = item['Question']
    choices = [item['Incorrect Answer 1'],item['Incorrect Answer 2'],item['Incorrect Answer 3'],item['Correct Answer']]
    cs_string = format_choices2str(choices)
    re_question = f"{question} Please select the letter corresponding to the answer you believe to be correct from the options below.\n{cs_string}"
    re_answer = str(item['Correct Answer'])  # 选项字母
    data_src = "GPQA"
    VGU_dataset.append(
        dict(zip(col_names,[re_question, re_answer, data_src]))
    )
# 组合
VGU_dataset = pd.DataFrame(VGU_dataset)
VGU_dataset = VGU_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
VGU_dataset.to_csv("./datasets/VGU_benchmark/annotation.csv", index=True)