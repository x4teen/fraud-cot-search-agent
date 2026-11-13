#!/usr/bin/env bash
# One-command demo: training snapshot + one batch of 4 test cases through the search agent.
# Usage: bash scripts/demo.sh [MODEL]     MODEL = Qwen/Qwen3-8B (GPU) or mock (no GPU)
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-8B}"
BACKEND=hf; [ "$MODEL" = "mock" ] && BACKEND=mock
TAG=$(echo "$MODEL" | sed -E 's/[^A-Za-z0-9]+/-/g; s/^-|-$//g')

echo "== 1. unit tests";             python -m pytest -q
echo "== 2. training snapshot";      python -m src.train_classical
ls results/checkpoints/B2_xgboost_raw28
echo "== 3. agent demo batch (4 cases, perf-first vs quality-first)"
python -m src.run_agent --backend "$BACKEND" --model "$MODEL" --n 4 \
       --conditions perf-agent qual-agent --verbose --run-tag demo
python -m src.show_trace "results/agent_${BACKEND}_${TAG}_test_demo.jsonl" --condition qual-agent --max 1
