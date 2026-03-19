import os, json, re
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

relation_save_dir = '/hongbojiang/codes/VGU/assets/relation'
os.makedirs(relation_save_dir, exist_ok=True)

def conditional_expectation_analysis(vgu_scores, render_scores, n_bins=5):
    """
    计算并可视化 Render 得分分箱下 VGU 的条件期望
    参数:
        vgu_scores: VGU 得分数组
        render_scores: Render 得分数组（0-1 连续值）
        n_bins: 分箱数量，默认5个等距分箱
    返回:
        分箱结果 DataFrame，包含每个分箱的 Render 区间、VGU 均值、样本量
    """
    # 1. 数据整理
    df = pd.DataFrame({
        'Render': render_scores,
        'VGU': vgu_scores
    }).dropna()  # 去除缺失值
    
    # 2. 对 Render 得分进行等距分箱（修复核心）
    # 生成等距分箱边界（n_bins个分箱 → n_bins+1个边界）
    bin_edges = np.linspace(df['Render'].min(), df['Render'].max(), n_bins + 1)
    # 生成准确的分箱标签（处理最后一个分箱的闭区间）
    bin_labels = []
    for i in range(n_bins):
        left = round(bin_edges[i], 1)
        right = round(bin_edges[i+1], 1)
        # 最后一个分箱用 [a, b]，其余用 [a, b)
        if i == n_bins - 1:
            bin_labels.append(f"[{left}, {right}]")
        else:
            bin_labels.append(f"[{left}, {right})")
    
    # 给数据打标签（指定bins和labels，避免自动生成的问题）
    df['Render_Bin'] = pd.cut(
        df['Render'],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True  # 包含最小值，避免第一个分箱漏样本
    )
    
    # 3. 计算每个分箱的条件期望（VGU 均值）和样本量
    result = df.groupby('Render_Bin', observed=False).agg(
        VGU_Mean=('VGU', 'mean'),
        Sample_Count=('VGU', 'count')
    ).reset_index()
    
    # 4. 可视化分箱均值
    plt.figure(figsize=(10, 6))
    plt.bar(result['Render_Bin'], result['VGU_Mean'], color='#1f77b4', alpha=0.7)
    plt.plot(result['Render_Bin'], result['VGU_Mean'], color='crimson', marker='o', linewidth=2, label='VGU mean')
    
    # 图表美化
    plt.xlabel('Render Score bin', fontsize=12)
    plt.ylabel('VGU conditional-expectation(mean)', fontsize=12)
    plt.title('Render vs VGU', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('/hongbojiang/codes/VGU/assets/relation/conditional_expectation.png', dpi=300, bbox_inches='tight')
    

def plot_lowless(vgu_scores, render_scores, save_path):
    """
    绘制 LOWESS 平滑曲线并保存
    参数：
        vgu_scores: 真实的 VGU 得分列表/数组
        render_scores: 真实的 Render 得分列表/数组
        save_path: 保存路径（如 "render_vgu_lowess.png"）
    """
    # ✅ 移除模拟数据，直接使用传入的真实数据
    # 合并为 DataFrame 方便处理（增加空值检查）
    df = pd.DataFrame({
        'Render': render_scores,
        'VGU': vgu_scores
    }).dropna()  # 去除空值，避免拟合报错

    # 检查数据是否为空
    if df.empty:
        print("错误：输入数据为空或全是缺失值！")
        return

    # LOWESS 拟合
    lowess_results = lowess(
        endog=df['VGU'],  # y轴：VGU score
        exog=df['Render'],  # x轴：Render score
        frac=0.2,  # 平滑窗口比例，可根据数据调整
        it=3
    )

    # 提取拟合后的 x 和 y
    x_fit = lowess_results[:, 0]
    y_fit = lowess_results[:, 1]

    # ✅ 显式创建新的图表对象，避免叠加
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制原始散点
    ax.scatter(df['Render'], df['VGU'], alpha=0.3, color='gray', label='data')

    # 绘制 LOWESS 平滑曲线
    ax.plot(x_fit, y_fit, color='crimson', linewidth=2.5, label='LOWESS')

    # 图表美化
    ax.set_xlabel('Render Score', fontsize=12)
    ax.set_ylabel('VGU Score', fontsize=12)
    ax.set_title('LOWESS Render vs VGU', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()  # 补充图例显示

    # ✅ 保存图表（核心步骤，无顺序问题，因为没用 plt.show()）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # ✅ 关闭图表对象，释放内存，避免循环调用时出错
    plt.close(fig)
    
    print(f"图表已成功保存到：{save_path}")

# ------------------- 6种独立阈值获取函数 -------------------
def get_median(scores):
    return sorted(scores)[len(scores)//2]

def get_threshold_mean(scores):
    """1. 均值法阈值"""
    scores = np.array(scores)
    return float(np.mean(scores))

def get_threshold_std_low(scores, k=1.0):
    """2. 低标准差倍数法 (均值 - k倍标准差) 宽松阈值"""
    scores = np.array(scores)
    return float(np.mean(scores) - k * np.std(scores, ddof=1))

def get_threshold_std_high(scores, k=1.0):
    """3. 高标准差倍数法 (均值 + k倍标准差) 严格阈值"""
    scores = np.array(scores)
    return float(np.mean(scores) + k * np.std(scores, ddof=1))

def get_threshold_quantile_low(scores):
    """4. 下四分位法阈值 (0.25分位数/Q1) 宽松阈值"""
    scores = np.array(scores)
    return float(np.percentile(scores, 25))

def get_threshold_quantile_high(scores):
    """5. 上四分位法阈值 (0.75分位数/Q3) 严格阈值"""
    scores = np.array(scores)
    return float(np.percentile(scores, 75))

# ------------------- 配套：成功/失败判定函数（极简版，必用） -------------------
def judge_score(scores, threshold):
    """得分≥阈值=成功(True)，得分<阈值=失败(False)，所有5个阈值通用该规则"""
    return [s >= threshold for s in scores]

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


def single_proc():
    idx = 12
    save_dir = "/hongbojiang/workdirs/COMSC_llm_judge"
    model_name = "BLIP3o"
    task_name = "VGU"
    json_path = os.path.join(save_dir, task_name, f"{model_name}.json")
    judge_dict = read_json_tostr(json_path)
    print(jsonstr_to_resultdict(judge_dict[idx]['judge_result'][0]))


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



from scipy.stats import spearmanr

# VGU和Render关系分析
def batch_proc1(model_name_list, threshold=1, sf_tau_func=get_median):

    def get_score(judge_dict):
        eval_result = jsonstr_to_scores(judge_dict['judge_result'][0])
        return (eval_result['legibility'] + eval_result['completeness'] + eval_result['correctness'])/3

    def recover_bymap(render_scores, mapping):
        target_num = sum([len(i) for i in mapping])
        output = list(range(target_num))
        for i in range(len(mapping)):
            target = render_scores[i]
            for idx in mapping[i]:
                output[idx] = target
        return output

    # 加载从VGU到Render的映射
    with open("/hongbojiang/datasets/VGU_benchmark/annotations/vgu2render_mapping.json", 'r') as f:
        mapping = json.load(f)
    ######################################################
    # 分开测试
    for model_name in model_name_list:
        # 获取得分
        vgu_scores = []
        render_scores = []
        vgu_judge_dict = read_json_tostr(os.path.join(save_dir, "VGU", f"{model_name}.json"))
        render_judge_dict = read_json_tostr(os.path.join(save_dir, "Render", f"{model_name}.json"))
        for i in range(len(vgu_judge_dict)):
            vgu_scores.append(get_score(vgu_judge_dict[i]))
            render_scores.append(get_score(render_judge_dict[mapping[str(i)]]))
        # 归一化
        vgu_scores = [i/5 for i in vgu_scores]
        render_scores = [i/5 for i in render_scores]
        # 计算每个模型中VGU和Render任务的相关系数
        # pearson相关系数和spearmanr系数
        pearson_corr, pearson_p = pearsonr(vgu_scores, render_scores)
        spearman_corr, spearman_p = spearmanr(vgu_scores, render_scores)
        print(f"{model_name}:")
        print(f"pearson r: {pearson_corr:.4f}")
        print(f"pearson p: {pearson_p:.4f}")
        print(f"Spearmanr rho: {spearman_corr:.4f}")
        print(f"Spearmanr p: {spearman_p:.4f}")
        # plot_lowless(vgu_scores, render_scores, os.path.join(relation_save_dir, f"{model_name}.jpg"))
        # # 计算条件概率
        # counts = [[0,0],[0,0]]
        # # 取中位数
        # threshold1 = threshold
        # threshold2 = threshold
        # # print(f"阈值在VGU序列中占比:{vgu_scores.count(threshold1)}/{len(vgu_scores)}.")
        # # print(f"阈值在VGU序列中占比:{render_scores.count(threshold2)}/{len(render_scores)}.")
        # for i in range(len(vgu_scores)):
        #     if vgu_scores[i]<=threshold1 and render_scores[i]<=threshold2:  # VF+RF
        #         counts[0][0]+=1
        #     elif vgu_scores[i]>threshold1 and render_scores[i]<=threshold2:    # VS+RF
        #         counts[0][1]+=1
        #     elif vgu_scores[i]<=threshold1 and render_scores[i]>threshold2:   # VF+RS
        #         counts[1][0]+=1
        #     elif vgu_scores[i]>threshold1 and render_scores[i]>threshold2:  # VS+RS
        #         counts[1][1]+=1
        # # 计算P(VF|RF)=RF&VF/RF(Render失败时，VGU失败可能性)
        # if counts[0][0]+counts[0][1]==0:
        #     p1 = 0
        # else:
        #     p1 = counts[0][0]/(counts[0][0]+counts[0][1])
        # print("P(VF|RF):", p1)
        # # 计算P(VF|RS)=RS&VF/RS(Render成功时，VGU失败可能性)
        # if counts[1][1]+counts[1][0]==0:
        #     p2 = 0
        # else:
        #     p2 = counts[1][0]/(counts[1][1]+counts[1][0])
        # print("P(VF|RS):",p2)
        # if p1!=0:
        #     mig = round(((p1-p2)/p1)*100, 2)
        # else:
        #     mig = None
        # print(f"缓解: {mig}%")
        # print("#"*30) 
    ######################################################
    # 综合测试    
    vgu_scores = []
    render_scores = []
    for model_name in model_name_list:
        vgu_judge_dict = read_json_tostr(os.path.join(save_dir, "VGU", f"{model_name}.json"))
        render_judge_dict = read_json_tostr(os.path.join(save_dir, "Render", f"{model_name}.json"))
        for i in range(len(vgu_judge_dict)):
            vgu_scores.append(get_score(vgu_judge_dict[i]))
            render_scores.append(get_score(render_judge_dict[mapping[str(i)]]))
    # 归一化
    vgu_scores = [i/5 for i in vgu_scores]
    render_scores = [i/5 for i in render_scores]
    # 计算每个模型中VGU和Render任务的相关系数
    # pearson相关系数和spearmanr系数
    pearson_corr, pearson_p = pearsonr(vgu_scores, render_scores)
    spearman_corr, spearman_p = spearmanr(vgu_scores, render_scores)
    print("综合测试:")
    print(f"pearson r: {pearson_corr:.4f}")
    print(f"pearson p: {pearson_p:.4f}")
    print(f"Spearmanr rho: {spearman_corr:.4f}")
    print(f"Spearmanr p: {spearman_p:.4f}")
    # plot_lowless(vgu_scores, render_scores, os.path.join(relation_save_dir, "all.jpg"))
    conditional_expectation_analysis(vgu_scores, render_scores)
    # # 计算条件概率
    # counts = [[0,0],[0,0]]
    # # 取中位数
    # threshold1 = threshold
    # threshold2 = threshold
    # for i in range(len(vgu_scores)):
    #     if vgu_scores[i]<=threshold1 and render_scores[i]<=threshold2:  # VF+RF
    #         counts[0][0]+=1
    #     elif vgu_scores[i]>threshold1 and render_scores[i]<=threshold2:    # VS+RF
    #         counts[0][1]+=1
    #     elif vgu_scores[i]<=threshold1 and render_scores[i]>threshold2:   # VF+RS
    #         counts[1][0]+=1
    #     elif vgu_scores[i]>threshold1 and render_scores[i]>threshold2:  # VS+RS
    #         counts[1][1]+=1
    # # 计算P(VF|RF)=RF&VF/RF(Render失败时，VGU失败可能性)
    # if counts[0][0]+counts[0][1]==0:
    #     p1 = 0
    # else:
    #     p1 = counts[0][0]/(counts[0][0]+counts[0][1])
    # print("P(VF|RF):", p1)
    # # 计算P(VF|RS)=RS&VF/RS(Render成功时，VGU失败可能性)
    # if counts[1][1]+counts[1][0]==0:
    #     p2 = 0
    # else:
    #     p2 = counts[1][0]/(counts[1][1]+counts[1][0])
    # print("P(VF|RS):", p2)
    # if p1!=0:
    #     mig = round(((p1-p2)/p1)*100, 2)
    # else:
    #     mig = None
    # print(f"缓解: {mig}%")
    # print("#"*30)
    return

if __name__=="__main__":
    # 三大维度评价指标整理
    # single_proc()

    # print("="*40)
    # print("VGU:")
    # save_dir = "/hongbojiang/workdirs/COMSC_llm_judge"
    # task_name = "VGU"
    # MODELS_TO_EVALUATE = [
    #     "Emu3",
    #     "Bagel",
    #     # "BLIP3o",
    #     # "Janus",
    #     # "JanusFlow", 
    #     # "Show-o",
    #     # "UniLIP",
    #     # "LongCat",
    #     # "Qwen-Image",
    #     # "VGT"
    # ]
    # model_eval_result = batch_proc0(MODELS_TO_EVALUATE, task_name, save_dir)
    # df = pd.DataFrame.from_dict(model_eval_result, orient='index')
    # df.to_excel(f"result_{task_name}.xlsx")
    # print(model_eval_result)
    ########################################################################
    # print("="*40)
    # print("Render:")
    # save_dir = "/hongbojiang/workdirs/COMSC_llm_judge"
    # task_name = "Render"
    # MODELS_TO_EVALUATE = [
    #     "Emu3",
    #     # "Bagel",
    #     # "BLIP3o",
    #     # "Janus",
    #     # "JanusFlow", 
    #     # "Show-o",
    #     # "UniLIP",
    #     # "LongCat",
    #     # "Qwen-Image",
    #     # "VGT"
    # ]
    # model_eval_result = batch_proc0(MODELS_TO_EVALUATE, task_name, save_dir)
    # df = pd.DataFrame.from_dict(model_eval_result, orient='index')
    # df.to_excel(f"result_{task_name}.xlsx")
    # print(model_eval_result)
    ########################################################################
    # print("="*40)
    # print("TGU:")
    # save_dir = "/hongbojiang/workdirs/COMSC_llm_judge"
    # task_name = "TGU"
    # MODELS_TO_EVALUATE = [
    #     "Bagel",
    #     "BLIP3o",
    #     "Emu3",
    #     "Janus",
    #     "JanusFlow", 
    #     "Show-o",
    #     "UniLIP",
    # ]
    # model_eval_result = batch_proc0(MODELS_TO_EVALUATE, task_name, save_dir)
    # df = pd.DataFrame.from_dict(model_eval_result, orient='index')
    # df.to_excel(f"result_{task_name}.xlsx")
    # print(model_eval_result)
    ########################################################################
    # UniModel中VGU和Render任务相关性分析
    MODELS_TO_EVALUATE = [
        "Bagel",
        "BLIP3o",
        "Emu3",
        "Janus",
        "JanusFlow", 
        "Show-o",
        "UniLIP",
        "LongCat",
        "Qwen-Image",
        "VGT"
    ]
    save_dir = "/hongbojiang/workdirs/COMSC_llm_judge"
    task_name="VGU"
    model_eval_result = batch_proc1(MODELS_TO_EVALUATE, 1, get_median)
    