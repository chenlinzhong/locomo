#!/bin/bash

config=$1
echo "📄 Using config: $config"

export HF_ENDPOINT=https://hf-mirror.com

max_retry=100

run_with_retry() {
    local cmd="$1"
    local attempt=1

    while [ $attempt -le $max_retry ]; do
        echo "➡️ Attempt $attempt: $cmd"
        eval "$cmd"
        status=$?
        if [ $status -eq 0 ]; then
            echo "✅ Success: $cmd"
            return 0
        else
            echo "❌ Failed: $cmd"
            attempt=$((attempt + 1))
            sleep 2
        fi
    done

    echo "🔥 Giving up after $max_retry attempts: $cmd"
    exit 1
}

# 逐步执行每个阶段，失败自动重试
run_with_retry "python json_to_nl.py --config ${config}"
run_with_retry "python get_answer_from_llm.py --config ${config}"
run_with_retry "python merge.py --config ${config}"
run_with_retry "python batch_score.py --config ${config}"
