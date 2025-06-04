#! /bin/bash
TOTAL_SIZE=15000

# math
MATH_SIZE=2500
python reasoners/tools/model_filtering/run_sample_and_postprocess.py \
    --input_data_dir data/train_filtered \
    --input_data_names math__merged \
    --output_data_dir data/train_guru15k \
    --target_sample_size $MATH_SIZE \
    --domain math > math_sample.log
