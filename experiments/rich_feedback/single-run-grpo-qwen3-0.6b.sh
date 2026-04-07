export USER=isaac
export TASK=datasets/lcb_v6
export PYTHONPATH=/home/isaac/SDPO:$PYTHONPATH
export WANDB_ENTITY=isaac-8080-io
export WANDB_MODE=online
export WANDB_API_KEY="$(cat /home/isaac/.wandb_api_key)"
export WANDB_CONSOLE=off
export VERL_REWARD_PROGRESS=1
export VERL_REWARD_PROGRESS_EVERY=1
export VERL_MICROBATCH_PROGRESS=1
export CUDA_VISIBLE_DEVICES=0
export N_GPUS_PER_NODE=1

python -m verl.trainer.main_ppo \
  --config-name baseline_grpo \
  data.train_files=/home/isaac/SDPO/datasets/lcb_v6/train.parquet \
  data.val_files=/home/isaac/SDPO/datasets/lcb_v6/test.parquet \
  data.train_batch_size=32 \
  trainer.group_name=GRPO-rich-feedback \
  trainer.experiment_name=local-grpo-qwen3-0.6b \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  reward_model.reward_manager=batch \
  algorithm.rollout_correction.rollout_is=token \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25
