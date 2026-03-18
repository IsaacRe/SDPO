#!/bin/bash
set -euo pipefail

cd /workspace/SDPO

export USER=root
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export XDG_CACHE_HOME=/workspace/.cache
export TMPDIR=/workspace/tmp
export RAY_TMPDIR=/workspace/ray_tmp

LOG_FILE=${1:-/workspace/tmp/chem_qwen3_8b_sdpo_paperlike.log}

bash training/verl_training.sh chem-sdpo-qwen3-8b-h200-paperlike sdpo datasets/sciknoweval/chemistry \
  trainer.logger=['console'] \
  vars.dir=/workspace/SDPO \
  vars.log_dir=/workspace/SDPO/output \
  vars.ckpt_dir=/workspace/SDPO/output/checkpoints \
  custom_reward_function.path=/workspace/SDPO/verl/utils/reward_score/feedback/__init__.py \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=120 \
  trainer.total_epochs=1 \
  trainer.test_freq=10 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.max_model_len=18944 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.logprobs_mode=null \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.use_remove_padding=False \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  max_model_len=18944 \
  > "$LOG_FILE" 2>&1
