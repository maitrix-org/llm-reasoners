```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/cot_gsm8k/inference_new.py --exllama_model_dir $LLAMA2_CKPTS --exllama_lora_dir None --exllama_mem_map '[16,22]' | tee cot_log.log
```
accuracy: 0.461
65/1319 got `None` output. A small portion of outputs are falling into loop, while more are of wrong formats, i.e., not ending with "the answer is ..."