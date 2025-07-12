import json
from datetime import datetime
import os
import argparse

def format_datetime(dt_str):
    """格式化为英文习惯的时间字符串"""
    dt = datetime.strptime(dt_str, "%I:%M %p on %d %B, %Y")
    return dt.strftime("%B %d, %Y at %I:%M %p")

def load_config(config_path):
    """从配置文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def get_output_dir(config, dataset_path):
    """获取输出目录路径"""
    # 优先使用配置文件中指定的output_dir
    output_dir = config.get('output_dir')
    
    if output_dir:
        # 如果output_dir是相对路径，则相对于配置文件所在目录解析
        if not os.path.isabs(output_dir):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            output_dir = os.path.join(config_dir, output_dir)
    else:
        # 默认使用数据集同目录下的locomo10_output
        output_dir = os.path.join(os.path.dirname(dataset_path), "locomo10_output")
    
    return output_dir

def convert_to_chat_and_qa_format(config_path):
    """将数据转换为对话和QA格式"""
    # 加载配置文件
    config = load_config(config_path)
    dataset_path = config.get('dataset')
    
    if not dataset_path:
        raise ValueError("配置文件中缺少'dataset'字段")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    # 获取输出目录
    output_dir = get_output_dir(config, dataset_path)
    
    # 加载数据集
    with open(dataset_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录设置为: {output_dir}")

    for i, sample in enumerate(data):
        output_lines = []
        
        # 添加对话信息
        output_lines.append("The following is the dialogue between the two people: {} and {} The dialogue lasted for several days, and the date of each dialogue was written at the beginning of the dialogue。\n".format(
            sample["conversation"]["speaker_a"],
            sample["conversation"]["speaker_b"]
        ))

        # 收集所有 session
        sessions = []
        for k in sample["conversation"].keys():
            if k.startswith("session_") and not k.endswith("_date_time"):
                session_key = k
                date_key = f"{k}_date_time"
                if date_key in sample["conversation"]:
                    session_time = format_datetime(sample["conversation"][date_key])
                    session_dialog = sample["conversation"][session_key]
                    sessions.append((session_time, session_dialog))

        # 按时间排序
        sessions.sort(key=lambda x: x[0])

        # 添加对话内容
        for dt, session in sessions:
            output_lines.append(dt)
            for msg in session:
                speaker = msg["speaker"]
                text = msg["text"]
                output_lines.append(f"{speaker}：{text}")

                if "img_url" in msg and msg["img_url"]:
                    caption = msg.get("blip_caption", "")
                    for url in msg["img_url"]:
                        output_lines.append(f"img({url}) ({caption})")
            output_lines.append("")

        # 添加问题回答部分
        output_lines.append("According to the above dialogue, write down the simplest answer with the least number of words to each of the following questions in a few sentences.")
        output_lines.append('Write the answer in the form of a json, where each entry contains the question number as "key" and the short answer as "value". ex: {{"0":"xx","1":"xx"}}')
        output_lines.append("Try to answer with the exact words in the dialogue.")

        qa_pairs = sample["qa"]
        answers = {}

        for idx, qa in enumerate(qa_pairs):
            category = qa.get('category')
            if qa.get('category') == 5:
                continue
            question = qa["question"]
            answer = qa["answer"]
            output_lines.append(f"{idx}:{question}")
            answers[str(idx)] = {
                "category": category,
                "answer": answer,
                "question":question,
            }

        # 写入输出文件
        output_file = os.path.join(output_dir, f"sample_{i}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

        # 写入答案文件
        answers_file = os.path.join(output_dir, f"answers_{i}.json")
        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='转换对话数据集格式')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径，包含dataset和output_dir字段')
    
    args = parser.parse_args()
    
    try:
        convert_to_chat_and_qa_format(args.config)
        print("转换完成！")
    except Exception as e:
        print(f"处理失败: {str(e)}")