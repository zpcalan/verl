import sys
import os
import numpy as np

good_inputs_f = "/data/k8s/zpc/Custom-LLaMA-Factory/zpc_script/good_inputs_rank.py.npy"
good_attn_mask_f = "/data/k8s/zpc/Custom-LLaMA-Factory/zpc_script/good_attention_mask.py.npy"
bad_inputs_f = "/data/k8s/zpc/Custom-LLaMA-Factory/zpc_script/bad_inputs_rank.py.npy"
bad_attn_mask_f = "/data/k8s/zpc/Custom-LLaMA-Factory/zpc_script/bad_attention_mask.py.npy"

good_inputs = np.load(good_inputs_f)
good_attn_mask = np.load(good_attn_mask_f)
bad_inputs = np.load(bad_inputs_f)
bad_attn_mask = np.load(bad_attn_mask_f)
for i in range(good_inputs.shape[0]):
    # import pdb; pdb.set_trace()
    inputs_not_151643 = good_inputs[i] != 151643
    good_inputs_not_151643 = inputs_not_151643.nonzero()
    print(f"good_inputs_not_151643 is  {good_inputs_not_151643[0][0]} {good_inputs_not_151643[0][-1]}")
for i in range(good_attn_mask.shape[0]):
    attn_mask_not_zero = good_attn_mask[i].nonzero()
    print(f"attn_mask_not_zero is {attn_mask_not_zero[0][0]} {attn_mask_not_zero[0][-1]}")

for i in range(bad_inputs.shape[0]):
    inputs_not_151643 = bad_inputs[i] != 151643
    bad_inputs_not_151643 = inputs_not_151643.nonzero()
    print(f"bad_inputs_not_151643 is {bad_inputs_not_151643[0][0]} {bad_inputs_not_151643[0][-1]}")
for i in range(bad_attn_mask.shape[0]):
    attn_mask_not_zero = bad_attn_mask[i].nonzero()
    print(f"attn_mask_not_zero is {attn_mask_not_zero[0][0]} {attn_mask_not_zero[0][-1]}")