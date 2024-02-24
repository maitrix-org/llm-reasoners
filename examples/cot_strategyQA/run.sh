export CUDA_VISIBLE_DEVICES=4,5
export EXLLAMA_CKPT=/home/shibo

log_dir='logs/cot_strategyqa-dev-70B-sc10'
mkdir -p $log_dir

nohup python examples/cot_strategyQA/inference.py --self_consistency 10 \
    --log_dir $log_dir --temperature 0.8 --resume 0 \
    > $log_dir/log.txt 2> $log_dir/nohup.out &