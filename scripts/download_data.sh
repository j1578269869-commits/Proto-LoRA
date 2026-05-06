#!/bin/bash
set -e
cd "$(dirname "$0")/.."

mkdir -p data/MET
mkdir -p data/MRE
mkdir -p outputs

echo "Data directories created."
echo "Please download or generate the required datasets and model prototypes into the following folders:"
echo "  data/MET/"
echo "  data/MRE/"
echo "  outputs/"
echo
cat <<'EOF'
Example expected structure:
  data/MET/train.json
  data/MET/valid.json
  data/MET/test.json
  data/MET/wikinewsImgs/

  data/MRE/mnre_llava_train.json
  data/MRE/mnre_llava_val.json
  data/MRE/mnre_llava_test.json
  data/MRE/image/

  data/MET/entity_embedding_idx.pt
  data/MET/entity_embedding_idx_llava.pt
  data/MET/entity_embedding_idx_Qwen2.pt
  data/MRE/extract_relation_embedding_blip.pt
  data/MRE/embeddings_similarity.pt
  data/MRE/similarity_embedding_Qwen2.pt
EOF

echo "If you have a dataset download link, add it to this script or use one of the config files in config/."