#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p ./dag_reasoning_reflex_log

DATA_DIR="$(pwd)/condqa_dataset"
DOCUMENTS_PATH="${DATA_DIR}/documents.json"
MODEL_NAME="qwen7b"
CONFIG_NAME="config_7b_rl"
TEMP="0.7"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -u dag_reasoning_with_reflexion.py \
    --model_name "${MODEL_NAME}" --config "${CONFIG_NAME}" --dataset dag --temp "${TEMP}" \
    --data_path "${DATA_DIR}/dag_test_split.json" --documents_path "${DOCUMENTS_PATH}" \
    > "./dag_reasoning_reflex_log/dag_qwen_7b.log"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -u dag_reasoning_with_reflexion.py \
    --model_name "${MODEL_NAME}" --config "${CONFIG_NAME}" --dataset condqa --temp "${TEMP}" \
    --data_path "${DATA_DIR}/condqa_test_split.json" --documents_path "${DOCUMENTS_PATH}" \
    > "./dag_reasoning_reflex_log/condqa_qwen_7b.log"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -u dag_reasoning_with_reflexion.py \
    --model_name "${MODEL_NAME}" --config "${CONFIG_NAME}" --dataset sharcqa --temp "${TEMP}" \
    --data_path "${DATA_DIR}/sharc_test_split.json" --documents_path "${DOCUMENTS_PATH}" \
    > "./dag_reasoning_reflex_log/sharcqa_qwen_7b.log"
