#!/bin/bash
export DATA_DIR="data/parquet/"

export train_files="[$(\
  ls "${DATA_DIR}"/*.parquet \
    | xargs -n1 basename \
    | sed "s|^|'${DATA_DIR}|;s|$|'|" \
    | paste -sd, -
)]"
echo "train_files = $train_files"
export test_dir="data/test/"
export test_files=$train_files
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

export WANDB_PROJECT="Reward-fn-import"
export WANDB_EXPERIMENT_NAME="local-${BASE_MODEL##*/}-$(date +%s)"

export VLLM_ATTENTION_BACKEND="XFORMERS"
export HYDRA_FULL_ERROR=1

# 4. Clean up existing Ray state
echo "Stopping any existing Ray cluster…"
ray stop || true
rm -rf /tmp/ray/ray_current_cluster

project_name='DAPO'
exp_name='DAPO-Qwen2.5-7B'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=1100
max_response_length=6000
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
max_num_gen_batches=10
train_prompt_bsz=16
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=2
train_prompt_mini_bsz=32

# Paths
MODEL_PATH=${BASE_MODEL}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

custom_reward_function="reward_score/__init__.py"
name="_default_compute_score"

# 7. Launch training script
echo "Launching training…"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$train_files \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    custom_reward_function.path=llm-reasoners/reasoners/reward/__init__.py \
    custom_reward_function.name=_default_compute_score \
    trainer.total_epochs=15 $@