#!/bin/bash
# Adversarial training iteration 2
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
set -x

export PYTHONUNBUFFERED=1

# Load checkpoint from iteration 1
MODEL_PATH=${MODEL_PATH:-"${CHECKPOINT_DIR}/isolde_iter1/global_step_15/actor/huggingface"}
DATA_DIR=${DATA_DIR:-"./data"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
RUN_NAME=isolde_iter2

export WANDB_MODE=offline

python3 -m verl.trainer.main \
    config=scripts/config/config_iter2.yaml \
    data.train_files="[${DATA_DIR}/isolde_iter2@train]" \
    data.val_files="[${DATA_DIR}/isolde_iter1@test]" \
    data.mini_rollout_batch_size=400 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=${CHECKPOINT_DIR}/${RUN_NAME} \
    worker.reward.reward_function=./scripts/reward/correct_first.py:compute_mcq_reward
