import torch
import os
from transformers import (
    AutoModelForTokenClassification,
)
from safetensors.torch import save_file

def convert_token_classifier_head_to_value_head(token_classifier_model_path):
    model = AutoModelForTokenClassification.from_pretrained(
            token_classifier_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
    state_dict = model.state_dict()
    value_weight = state_dict['score.weight']
    value_bias = state_dict['score.bias']
    value_head_dict = {
        'v_head.summary.weight': value_weight,
        'v_head.summary.bias': value_bias,
    }
    vhead_file = os.path.join(token_classifier_model_path, "value_head.safetensors")
    save_file(value_head_dict, vhead_file)
convert_token_classifier_head_to_value_head("/data/k8s/zpc/Custom-LLaMA-Factory/arkts_linter_reward_model_merged_v1")