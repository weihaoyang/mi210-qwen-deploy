#!/usr/bin/env bash
set -euo pipefail
: "${LLAMA_BIN:?Set LLAMA_BIN to the llama-server executable}"
: "${MODEL:?Set MODEL to a compatible GGUF}"
export HIP_VISIBLE_DEVICES="${GPU:-0}"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_BIN"):${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
args=(-m "$MODEL" --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
      -ngl 99 -c "${CTX_TOTAL:-65536}" -np "${SLOTS:-4}"
      --jinja --no-webui --alias "${MODEL_ALIAS:-qwen3.8-27b}")
if [[ -n ${MMPROJ:-} ]]; then
  args+=(--mmproj "$MMPROJ")
fi
if [[ ${MTP:-1} == 1 ]]; then
  args+=(--spec-type draft-mtp --spec-draft-n-max "${DRAFT_MAX:-3}")
  if [[ -n ${DRAFT_MODEL:-} ]]; then
    args+=(--spec-draft-model "$DRAFT_MODEL" --spec-draft-ngl 99)
  fi
fi
exec "$LLAMA_BIN" "${args[@]}"
