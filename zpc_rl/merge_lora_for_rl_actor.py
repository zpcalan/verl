from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from peft import PeftModel
import torch
from safetensors import safe_open


lora_adapter_path = "/data/k8s/zpc/verl/zpc_rl/ckpt/global_step_350/actor/lora_adapter"
sft_model_path = "/data/k8s/zpc/Custom-LLaMA-Factory/arkts_linter_model_best/"

# 1. 加载 base model 并合并 LoRA
pretrained_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True)
model_after_lora = PeftModel.from_pretrained(pretrained_model, lora_adapter_path).merge_and_unload()


# 5. 保存模型（直接保存，避免 AutoModelForCausalLMWithValueHead 的额外开销）
merged_rl_model_path = "/data/k8s/zpc/Custom-LLaMA-Factory/arkts_linter_after_7_epoch_rl"
model_after_lora.save_pretrained(merged_rl_model_path)
tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
tokenizer.save_pretrained(merged_rl_model_path)

print(f"模型已保存到: {merged_rl_model_path}")