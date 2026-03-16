#!/bin/bash
set -euo pipefail

# Reproduces a scaled SDPO-vs-GRPO chemistry comparison on a single H200.
# Defaults are intentionally small enough for local execution.

export USER="${USER:-root}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_PATH="${DATA_PATH:-datasets/sciknoweval/chemistry}"
STEPS="${STEPS:-8}"
TRAIN_BS="${TRAIN_BS:-4}"
ROLLOUT_N="${ROLLOUT_N:-2}"
VAL_N="${VAL_N:-4}"
LOG_DIR="${LOG_DIR:-/tmp}"
TS="$(date +%Y%m%d_%H%M%S)"

SDPO_LOG="$LOG_DIR/chem_sdpo_${TS}.log"
GRPO_LOG="$LOG_DIR/chem_grpo_${TS}.log"

if [ ! -f "$DATA_PATH/train.parquet" ] || [ ! -f "$DATA_PATH/test.parquet" ]; then
  echo "[prep] parquet not found in $DATA_PATH; preprocessing"
  python data/preprocess.py --data_source "$DATA_PATH"
fi

COMMON_ARGS=(
  "trainer.logger=['console']"
  "vars.dir=$PROJECT_ROOT"
  "vars.log_dir=$PROJECT_ROOT/output"
  "vars.ckpt_dir=$PROJECT_ROOT/output/checkpoints"
  "custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"
  "trainer.n_gpus_per_node=1"
  "trainer.nnodes=1"
  "trainer.total_training_steps=$STEPS"
  "trainer.total_epochs=1"
  "trainer.test_freq=4"
  "trainer.val_before_train=False"
  "trainer.resume_mode=disable"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.35"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.logprobs_mode=null"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "+actor_rollout_ref.model.override_config.attn_implementation=eager"
  "actor_rollout_ref.actor.ppo_mini_batch_size=4"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.optim.lr=1e-5"
  "actor_rollout_ref.actor.use_remove_padding=False"
  "data.train_batch_size=$TRAIN_BS"
  "data.max_prompt_length=1024"
  "data.max_response_length=512"
  "max_model_len=2048"
)

echo "[run] SDPO -> $SDPO_LOG"
bash training/verl_training.sh chem-sdpo-h200 sdpo "$DATA_PATH" "${COMMON_ARGS[@]}" >"$SDPO_LOG" 2>&1

echo "[run] GRPO -> $GRPO_LOG"
bash training/verl_training.sh chem-grpo-h200 baseline_grpo "$DATA_PATH" "${COMMON_ARGS[@]}" >"$GRPO_LOG" 2>&1

extract_best_metric() {
  local file="$1"
  local key="$2"
  rg "$key" "$file" \
    | sed -E "s/.*step:([0-9]+).*${key}:np\\.float64\\(([0-9eE+\\.-]+)\\).*/\\1 \\2/" \
    | awk 'BEGIN{best=-1; best_step=-1} {if ($2+0 > best) {best=$2+0; best_step=$1}} END {if (best_step<0) print "NA NA"; else printf "%s %.12f", best_step, best}'
}

read -r sdpo_step sdpo_acc < <(extract_best_metric "$SDPO_LOG" "val-core/sciknoweval/acc/mean@4")
read -r grpo_step grpo_acc < <(extract_best_metric "$GRPO_LOG" "val-core/sciknoweval/acc/mean@4")
read -r sdpo_best_step sdpo_best_acc < <(extract_best_metric "$SDPO_LOG" "val-core/sciknoweval/acc/best@4/mean")
read -r grpo_best_step grpo_best_acc < <(extract_best_metric "$GRPO_LOG" "val-core/sciknoweval/acc/best@4/mean")

echo "[summary]"
echo "  SDPO peak acc/mean@4:      step=$sdpo_step value=$sdpo_acc"
echo "  GRPO peak acc/mean@4:      step=$grpo_step value=$grpo_acc"
echo "  SDPO peak acc/best@4/mean: step=$sdpo_best_step value=$sdpo_best_acc"
echo "  GRPO peak acc/best@4/mean: step=$grpo_best_step value=$grpo_best_acc"
echo "  Logs:"
echo "    $SDPO_LOG"
echo "    $GRPO_LOG"
