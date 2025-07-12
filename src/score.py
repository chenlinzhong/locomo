import json
import requests
import argparse
from collections import defaultdict
from metrics import calculate_metrics, calculate_bleu_scores
import pandas as pd
from tqdm import tqdm
import openai
import time
from metrics import send_to_llm
import os
import random


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_llm_judge(question, gt_answer, pred_answer, model, api_url, api_key, is_azure_openai):
    prompt = f"""你是一个问答判定助手，请根据以下内容判断“模型回答”是否正确。只返回“1”表示正确，“0”表示错误，不要有其它解释。

            问题：{question}
            参考答案：{gt_answer}
            模型回答：{pred_answer}

            请判断模型回答是否正确（1表示正确，0表示错误）：
    """

    try:
        content = send_to_llm(prompt, api_key, api_url, model, is_azure_openai)
        content = content["choices"][0]["message"]["content"].strip()
        # 只保留 "0" 或 "1"
        if content.startswith("1"):
            return "1"
        elif content.startswith("0"):
            return "0"
        else:
            print(f"⚠️ 未知判定输出：{content}")
            return None

    except Exception as e:
        print(f"❌ 请求 LLM 判定出错: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='配置文件路径')
    args = parser.parse_args()

    config = load_config(args.config)

    model = config["model"]
    api_url = config["openai_base_url"]
    api_key = config["api_key"]
    output_dir = config["output_dir"]

    input_path = output_dir.rstrip("/") + "/merged_answers_all.json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    local_results= defaultdict(list)

    output_file = f"{output_dir}/local_item_results.txt"

    # ✅ 尝试读取已保存的中间结果
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            local_results = json.load(f)
    else:
        local_results = {}

    for k, item in tqdm(data.items(), desc="Evaluating"):
        if "question" not in item or "answer" not in item or "response" not in item or "category" not in item:
            print(f"⚠️ 跳过不完整项: {k}")
            continue

        question = item["question"]
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = item["category"]

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)

        # ✅ 如果已有 llm_score，则复用
        if k in local_results and "llm_score" in local_results[k]:
            llm_score = local_results[k]["llm_score"]
        else:
            llm_score = evaluate_llm_judge(question, gt_answer, pred_answer, model, api_url, api_key, config['is_azure_openai'])
            time.sleep(2)
        local_results[k] = {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
            "llm_score": llm_score,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(local_results, f, indent=4, ensure_ascii=False)


    all_items = list(local_results.values())
    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])

    result = df.groupby("category").agg({
        "bleu_score": "mean",
        "f1_score": "mean",
        "llm_score": "mean"
    }).round(4)
    result["count"] = df.groupby("category").size()

    overall_means = df[["bleu_score", "f1_score", "llm_score"]].mean().round(4)

    output_path = f"{output_dir}/{model}_score.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Mean Scores Per Category:\n")
        f.write(result.to_string())
        f.write("\n\nOverall Mean Scores:\n")
        f.write(overall_means.to_string())


if __name__ == "__main__":
    main()
