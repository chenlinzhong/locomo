import json
import argparse
from pathlib import Path
import re

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_predictions(pred_file):
    with open(pred_file, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
        content = pred_data["choices"][0]["message"]["content"]
        content = content.replace('\\\"', '')
        # 尝试提取 JSON 内容
        try:
            # 清理大模型输出格式
            matches = re.findall(r'\{[\s\S]*?\}', content)
            if not matches:
                raise ValueError("No JSON object found in content")
            content = matches[0]
            return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[extract_predictions] Failed to parse JSON from file: {pred_file}")
            print(f"Error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file containing output_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])

    answer_files = sorted(output_dir.glob("answers_*.json"))
    prediction_files = sorted(output_dir.glob("prediction_*.json"))

    def extract_number(f): return int(re.findall(r"\d+", f.name)[0])
    answer_map = {extract_number(f): f for f in answer_files}
    prediction_map = {extract_number(f): f for f in prediction_files}

    merged_all = {}
    global_idx = 0

    for idx in sorted(set(answer_map.keys()) & set(prediction_map.keys())):
        answer_file = answer_map[idx]
        prediction_file = prediction_map[idx]

        with open(answer_file, "r", encoding="utf-8") as f:
            answers = json.load(f)
        predictions = extract_predictions(prediction_file)

        for key in answers:
            entry = answers[key]
            if key in predictions:
                entry["response"] = predictions[key]
            merged_all[str(global_idx)] = entry
            global_idx += 1

    output_file = output_dir / "merged_answers_all.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_all, f, ensure_ascii=False, indent=2)

    print(f"✅ 合并完成，共 {global_idx} 条，输出文件: {output_file}")

if __name__ == "__main__":
    main()
