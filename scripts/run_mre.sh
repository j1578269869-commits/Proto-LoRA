#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== Run MRE experiments with config files ====="

python trainlora_blip2_MRE.py --config config/mre_blip2.yaml
python trainlora_llava_MRE.py --config config/mre_llava.yaml
python trainlora_Qwen2_MRE.py --config config/mre_qwen.yaml
