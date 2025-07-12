import requests
import json
import os
import uuid
import glob
import argparse
from time import sleep
from tqdm import tqdm
import openai
from metrics import send_to_llm


def load_config(config_path):
    """从配置文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def process_all_samples(config, max_retries=3):
    """处理目录下所有sample_*.txt文件"""
    # 从配置获取参数
    directory = config.get('output_dir')
    api_key = config.get('api_key')
    api_url = config.get('openai_base_url')
    model_name = config.get('model', 'volc-deepseek-v3')  # 默认模型
    
    # 参数检查
    if not all([directory, api_key, api_url]):
        raise ValueError("配置文件中缺少必要的参数(output_dir, api_key, openai_base_url)")
    
    # 查找所有匹配的文件
    file_pattern = os.path.join(directory, "sample_*.txt")
    sample_files = sorted(glob.glob(file_pattern))
    
    if not sample_files:
        print(f"在目录 {directory} 中未找到 sample_*.txt 文件")
        return
    
    print(f"找到 {len(sample_files)} 个文件需要处理")
    print(f"使用模型: {model_name}")
    
    # 创建预测结果目录
    predictions_dir = directory
    os.makedirs(predictions_dir, exist_ok=True)
    
    for file_path in tqdm(sample_files, desc="预测答案"):
        filename = os.path.basename(file_path)
        sample_id = filename.split('_')[1].split('.')[0]
        output_file = os.path.join(predictions_dir, f"prediction_{sample_id}.json")
        
        # 如果文件已存在，跳过处理
        if os.path.exists(output_file):
            print(f"预测文件 {output_file} 已存在，跳过")
            continue
            
        # 重试机制
        result = None
        for attempt in range(max_retries):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                result = send_to_llm(content, api_key, api_url, model_name, config['is_azure_openai'])
                if result:
                    # 保存到独立文件
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"成功处理并保存 {output_file}")
                    break
                else:
                    print(f"第 {attempt + 1} 次尝试失败")
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试出错: {str(e)}")
            
            if attempt < max_retries - 1:
                sleep(2)  # 等待2秒后重试
        
        if not result:
            print(f"无法处理文件 {filename}，跳过")
    
    print(f"\n处理完成! 所有预测结果已保存到 {predictions_dir} 目录")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='处理对话样本并调用LLM API')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径，需包含output_dir, api_key和openai_base_url字段')
    
    args = parser.parse_args()
    
    try:
        # 加载配置文件
        config = load_config(args.config)
        
        # 处理所有文件
        process_all_samples(config)
    except Exception as e:
        print(f"处理失败: {str(e)}")