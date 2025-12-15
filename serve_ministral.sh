#!/usr/bin/env bash
set -euo pipefail

SIZE="${1:-8}"          # 14 | 8 | 3
Q="${2:-awq4}"           # awq4 | awq8 | fp8
PORT="${3:-8000}"
CTX="${4:-8192}"         # max context
SEQ="${5:-2}"            # concurrency
OFFLOAD="${6:-0}"        # CPUオフロードGiB
VARIANT="${7:-instruct}" # instruct | reasoning
KV_DTYPE="${8:-fp8}"     # auto|fp8|fp8_e5m2|fp8_e4m3fn|half
DTYPE="${9:-half}"       # auto|half|bfloat16
GPU_UTIL="${10:-0.80}"   # 0.80~0.95 目安
SWAP="${11:-8}"          # 0~16 目安（速度犠牲で安定）

# モデル解決
MODEL=""
QUANT_FLAG=""
case "$VARIANT" in
  instruct)
    case "$SIZE:$Q" in
      "14:awq4") MODEL="cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit";  QUANT_FLAG="--quantization compressed-tensors" ;;
      "14:awq8") MODEL="cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-8bit";  QUANT_FLAG="--quantization compressed-tensors" ;;
      "14:fp8")  MODEL="mistralai/Ministral-3-14B-Instruct-2512";          QUANT_FLAG="" ;;
      "8:awq4")  MODEL="cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit";   QUANT_FLAG="--quantization compressed-tensors" ;;
      "8:awq8")  MODEL="cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-8bit";   QUANT_FLAG="--quantization compressed-tensors" ;;
      "8:fp8")   MODEL="mistralai/Ministral-3-8B-Instruct-2512";           QUANT_FLAG="" ;;
      "3:awq4")  MODEL="cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit";   QUANT_FLAG="--quantization compressed-tensors" ;;
      "3:fp8")   MODEL="mistralai/Ministral-3-3B-Instruct-2512";           QUANT_FLAG="" ;;
      *) echo "Usage: $0 {14|8|3} {awq4|awq8|fp8} [PORT] [CTX] [SEQ] [OFFLOAD] [VARIANT] [KV_DTYPE] [DTYPE] [GPU_UTIL] [SWAP]"; exit 1 ;;
    esac
    ;;
  reasoning)
    case "$SIZE:$Q" in
      "14:awq4") MODEL="cyankiwi/Ministral-3-14B-Reasoning-2512-AWQ-4bit"; QUANT_FLAG="--quantization compressed-tensors" ;;
      "14:awq8") MODEL="cyankiwi/Ministral-3-14B-Reasoning-2512-AWQ-8bit"; QUANT_FLAG="--quantization compressed-tensors" ;;
      "14:fp8")  MODEL="mistralai/Ministral-3-14B-Reasoning-2512";         QUANT_FLAG="" ;;
      "8:awq4")  MODEL="cyankiwi/Ministral-3-8B-Reasoning-2512-AWQ-4bit";  QUANT_FLAG="--quantization compressed-tensors" ;;
      "8:awq8")  MODEL="cyankiwi/Ministral-3-8B-Reasoning-2512-AWQ-8bit";  QUANT_FLAG="--quantization compressed-tensors" ;;
      "8:fp8")   MODEL="mistralai/Ministral-3-8B-Reasoning-2512";          QUANT_FLAG="" ;;
      "3:awq4")  MODEL="cyankiwi/Ministral-3-3B-Reasoning-2512-AWQ-4bit";  QUANT_FLAG="--quantization compressed-tensors" ;;
      "3:fp8")   MODEL="mistralai/Ministral-3-3B-Reasoning-2512";          QUANT_FLAG="" ;;
      *) echo "Usage: $0 {14|8|3} {awq4|awq8|fp8} [PORT] [CTX] [SEQ] [OFFLOAD] [VARIANT] [KV_DTYPE] [DTYPE] [GPU_UTIL] [SWAP]"; exit 1 ;;
    esac
    ;;
  *)
    echo "VARIANT must be 'instruct' or 'reasoning'"; exit 1 ;;
esac

# 画像は1枚に制限（必要に応じ変更）
LIMIT_MM='{"image":1}'

CMD=(
vllm serve "${MODEL}"
  --host 0.0.0.0 --port "${PORT}"
  --tokenizer_mode mistral
  --config_format mistral
  --load_format mistral
  --max-model-len "${CTX}"
  --max-num-seqs "${SEQ}"
  --kv-cache-dtype "${KV_DTYPE}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --limit-mm-per-prompt "${LIMIT_MM}"
  --mm-processor-cache-gb 0
  --cpu-offload-gb "${OFFLOAD}"
  --swap-space "${SWAP}"
)

# Reasoning フラグ
if [[ "${VARIANT}" == "reasoning" ]]; then
  CMD+=( --reasoning-parser mistral )
fi

# 量子化指定（必要時のみ付与）
if [[ -n "${QUANT_FLAG}" ]]; then
  CMD+=(${QUANT_FLAG})
fi

echo "${CMD[@]}"
exec "${CMD[@]}"
