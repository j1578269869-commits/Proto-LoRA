#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "===== Run MET experiments with config files ====="

python trainlora_blip2_MET.py --config config/met_blip2.yaml
python trainlora_llava_MET.py --config config/met_llava.yaml
python trainlora_Qwen_MET.py --config config/met_qwen.yaml
