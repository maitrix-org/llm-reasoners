import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

def load_model(args, device):
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                trust_remote_code=True,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)

        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)
        model.to(device)

    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, add_bos_token=False, padding_side='left')


    if args.use_lora and not args.test_only:
        if "Phi" in args.pretrained_model or "Phi".lower() in args.pretrained_model:
            lora_config = LoraConfig(
            r=16, #Rank
            lora_alpha=32,
            target_modules=[
                 'q_proj',
            'k_proj',
            'v_proj',
            'dense'
            ],
            
            bias="none",
            lora_dropout=0.1,  # Conventional
            task_type="CAUSAL_LM",
        )
        else:
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "v_proj", "k_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, lora_config)
        return model, tokenizer
    if args.test_only and args.load_checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, args.load_checkpoint_path)
    return model, tokenizer