import json
import requests
import argparse
from collections import defaultdict
from metrics import calculate_metrics, calculate_bleu_scores, send_to_llm
import pandas as pd
from tqdm import tqdm
import time
import os
import re


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_batch_prompt(items):
    header = (
        "你是一个问答判定助手，请对多个问答对进行判断。"
        "我们会给出每个问题的编号、问题内容、参考答案和模型回答。"
        "你的任务是判断模型回答是否正确，结果用 JSON 格式返回，键是问题编号，值是 1（正确）或 0（错误）。"
        "不要添加多余解释，只返回严格符合 JSON 格式的内容。\n\n"
    )
    blocks = []
    for item_id, question, ref_answer, model_answer in items:
        block = (
            f"编号：{item_id}\n"
            f"问题：{question}\n"
            f"参考答案：{ref_answer}\n"
            f"模型回答：{model_answer}\n"
        )
        blocks.append(block)
    return header + "\n".join(blocks) + '\n请你返回如下格式的 JSON，例如：{"1":1,"2":0}'


def evaluate_llm_judge_batch_with_id(items, model, api_url, api_key, is_azure_openai):
    prompt = build_batch_prompt(items)
    try:
        response = send_to_llm(prompt, api_key, api_url, model, is_azure_openai)
        content = response["choices"][0]["message"]["content"]
        content = content.strip().strip("```json").strip("```").strip()
        matches = re.findall(r'\{[\s\S]*?\}', content)
        content = matches[0]
        result = json.loads(content.strip())
        return result
    except Exception as e:
        print(f"❌ 批量打分出错: {e}, {response=}")
        return {item[0]: None for item in items}  # 返回对应长度的空结果


def convert_scores_to_numeric_or_raise(local_results):
    for k, v in local_results.items():
        try:
            v["bleu_score"] = float(v["bleu_score"])
        except Exception as e:
            print(f"Key {k} 的 bleu_score 转 float 失败，值为: {v['bleu_score']}，错误: {e}")

        try:
            v["f1_score"] = float(v["f1_score"])
        except Exception as e:
            print(f"Key {k} 的 f1_score 转 float 失败，值为: {v['f1_score']}，错误: {e}")

        try:
            v["llm_score"] = int(v["llm_score"])
        except Exception as e:
            print(f"Key {k} 的 llm_score 转 int 失败，值为: {v['llm_score']}，错误: {e}")
            v["bleu_score"] = 0

    return local_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='配置文件路径')
    args = parser.parse_args()

    config = load_config(args.config)

    model = config["model"]
    api_url = config["openai_base_url"]
    api_key = config["api_key"]
    output_dir = config["output_dir"]
    batch_size = config.get("batch_size", 10)
    is_azure_openai = config.get("is_azure_openai", 1)

    input_path = output_dir.rstrip("/") + "/merged_answers_all.json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_file = f"{output_dir}/local_item_results.txt"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            local_results = json.load(f)
    else:
        local_results = {}

    keys = list(data.keys())
    total = len(keys)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch_keys = keys[i:i + batch_size]
        batch_items = []
        for k in batch_keys:
            item = data[k]
            if (
                "question" not in item or "answer" not in item or
                "response" not in item or "category" not in item
            ):
                print(f"⚠️ 跳过不完整项: {k}")
                continue
            if k in local_results and "llm_score" in local_results[k]:
                continue
            batch_items.append((k, item["question"], str(item["answer"]), str(item["response"])))

        if not batch_items:
            continue

        llm_scores = evaluate_llm_judge_batch_with_id(batch_items, model, api_url, api_key, is_azure_openai)
        time.sleep(2)
        for (k, question, gt_answer, pred_answer) in batch_items:
            item = data[k]
            metrics = calculate_metrics(pred_answer, gt_answer)
            bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
            llm_score = llm_scores.get(k, None)
            if llm_score is None:
                print(f"[llm_score missing] key: {k}")
                llm_score = 0

            local_results[k] = {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": item["category"],
                "bleu_score": float(bleu_scores["bleu1"]),
                "f1_score": float(metrics["f1"]),
                "llm_score": int(llm_score),
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(local_results, f, indent=4, ensure_ascii=False)

    # 汇总输出
    local_results = convert_scores_to_numeric_or_raise(local_results)
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
