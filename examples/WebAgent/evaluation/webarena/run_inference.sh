#!/bin/bash
set -x

BASE_URL="[your_host_name]"

export SHOPPING="$BASE_URL:7770/"
export SHOPPING_ADMIN="$BASE_URL:7780/admin"
export REDDIT="$BASE_URL:9999"
export GITLAB="$BASE_URL:8023"
export WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export MAP="$BASE_URL:3000"
export HOMEPAGE="$BASE_URL:4399"

cd ../..

# Note that after running a set of experiments with each agent, the sites 
# must be reset before running the next one following the guide:
# https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#environment-reset

python main.py webarena \
    --agent reasoner \
    --output_dir evaluation/webarena/results/baseline \
    --model gpt-4o \
    --dataset webarena \
    --config_name browsergym_webarena \
    --end_idx 100 \
    --shuffle \
    --max_steps 15 

# python main.py webarena \
#     --agent reasoner \
#     --output_dir evaluation/webarena/results/wmp \
#     --model gpt-4o \
#     --dataset webarena \
#     --config_name browsergym_webarena_world_model \
#     --end_idx 100 \
#     --shuffle \
#     --max_steps 15 

# python main.py webarena \
#     --agent openhands \
#     --output_dir evaluation/webarena/browsingagent \
#     --model gpt-4o \
#     --dataset webarena \
#     --config_name opendevin_webarena \
#     --end_idx 100 \
#     --shuffle \
#     --max_steps 15 
